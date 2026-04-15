from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import yaml


logger = logging.getLogger(__name__)


class Gripper:
    """UDP client for remote gripper control via a JSON protocol."""

    GRIPPER_DIRECTION = False

    def __init__(self) -> None:
        host = None
        port = None
        timeout = None

        config_candidates = [
            Path(__file__).resolve().parent / "config.yaml",
            Path("/home/hyunjin/rby1_ws/rby1-data-collection/config.yaml"),
        ]
        for config_path in config_candidates:
            try:
                if not config_path.exists():
                    continue
                with config_path.open(encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                host = cfg.get("remote_gripper_host", host)
                port = cfg.get("remote_gripper_port", port)
                timeout = cfg.get("remote_gripper_timeout", timeout)
                break
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load gripper config: %s", config_path)

        host = os.getenv("REMOTE_GRIPPER_HOST", host)
        port_env = os.getenv("REMOTE_GRIPPER_PORT")
        timeout_env = os.getenv("REMOTE_GRIPPER_TIMEOUT")

        if port_env is not None:
            try:
                port = int(port_env)
            except ValueError:
                logger.warning("Invalid REMOTE_GRIPPER_PORT: %s", port_env)

        if timeout_env is not None:
            try:
                timeout = float(timeout_env)
            except ValueError:
                logger.warning("Invalid REMOTE_GRIPPER_TIMEOUT: %s", timeout_env)

        self.host = host
        self.port = port
        self.timeout = timeout
        self.target_q: np.ndarray = np.zeros(2, dtype=float)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock_lock = threading.Lock()
        if self.timeout is not None:
            try:
                self._sock.settimeout(float(self.timeout))
            except Exception:  # noqa: BLE001
                pass

    def _udp_request(self, payload: dict, expect_reply: bool) -> dict | None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if self.host is None or self.port is None:
            logger.warning("Gripper host/port not configured")
            return None

        try:
            with self._sock_lock:
                self._sock.sendto(data, (self.host, self.port))
                if not expect_reply:
                    return None

                expected_cmd = payload.get("cmd")
                try:
                    sock_timeout = self._sock.gettimeout()
                except Exception:  # noqa: BLE001
                    sock_timeout = None
                if sock_timeout is None:
                    sock_timeout = 1.0

                deadline = time.monotonic() + float(sock_timeout)
                # Some servers may return cmd-less replies for older commands.
                # Keep one as fallback, but prefer exact cmd matches.
                expected_response = None
                while time.monotonic() < deadline:
                    remaining = max(0.0, deadline - time.monotonic())
                    self._sock.settimeout(remaining)
                    try:
                        raw, _ = self._sock.recvfrom(65535)
                    except socket.timeout:
                        break

                    try:
                        response = json.loads(raw.decode("utf-8"))
                    except Exception:  # noqa: BLE001
                        continue

                    if not isinstance(response, dict):
                        continue
                    response_cmd = response.get("cmd")
                    if expected_cmd is None:
                        return response

                    if response_cmd == expected_cmd:
                        return response
                    if response_cmd is None and expected_response is None:
                        expected_response = response

                return expected_response
        except Exception:  # noqa: BLE001
            return None

        return None

    def initialize(self, verbose: bool = False) -> bool:
        self._udp_request({"cmd": "initialize", "ts": time.time()}, expect_reply=False)
        response = self._udp_request({"cmd": "ping", "ts": time.time()}, expect_reply=True)
        ok = bool(response and response.get("ok", False))
        if verbose:
            logger.info("[Gripper] ping (%s:%s) -> %s", self.host, self.port, ok)
        return ok

    def set_operating_mode(self, mode: str) -> None:
        response = self._udp_request(
            {"cmd": "set_operating_mode", "mode": mode, "ts": time.time()}, expect_reply=True
        )
        if not response or not response.get("ok", False):
            raise RuntimeError("Failed to set remote operating mode")

    def homing(self) -> bool:
        response = self._udp_request({"cmd": "homing", "ts": time.time()}, expect_reply=True)
        if not isinstance(response, dict):
            logger.warning("[Gripper] homing failed or no response")
            return False

        ok = bool(response.get("ok", False))
        if not ok:
            logger.warning("[Gripper] homing failed or no response")
            return False
        self.min_q = np.asarray(response.get("min_q", None), dtype=float).reshape(-1)
        self.max_q = np.asarray(response.get("max_q", None), dtype=float).reshape(-1)
        logger.info("[Gripper] homing success. min_q=%s max_q=%s", self.min_q, self.max_q)
        return True

    def start(self) -> None:
        response = self._udp_request({"cmd": "start", "ts": time.time()}, expect_reply=True)
        if not response or not response.get("ok", False):
            raise RuntimeError("Failed to start remote gripper loop")

    def stop(self) -> None:
        response = self._udp_request({"cmd": "stop", "ts": time.time()}, expect_reply=True)
        if not response or not response.get("ok", False):
            raise RuntimeError("Failed to stop remote gripper loop")

    def get_target(self) -> np.ndarray:
        return self.target_q

    def get_normalized_target(self) -> np.ndarray:
        response = self._udp_request({"cmd": "get_normalized_target", "ts": time.time()}, expect_reply=True)
        if not response or not response.get("ok", False):
            raise RuntimeError("Failed to fetch normalized target")
        target = response.get("target", None)
        if target is None:
            raise RuntimeError("Normalized target missing in response")
        return np.asarray(target, dtype=float).reshape(-1)

    def set_normalized_target(self, normalized_q: np.ndarray, wait_for_reply: bool = False) -> None:
        normalized_q = np.asarray(normalized_q, dtype=float).reshape(-1)
        self.target_q = normalized_q
        response = self._udp_request(
            {
                "cmd": "set_normalized_target",
                "normalized_q": normalized_q.tolist(),
                "ts": time.time(),
            },
            expect_reply=bool(wait_for_reply),
        )
        if wait_for_reply:
            if not response or not response.get("ok", False):
                raise RuntimeError("Failed to set normalized target")
            if response.get("target", None) is not None:
                self.target_q = np.asarray(response.get("target", None), dtype=float).reshape(-1)

    def get_state(self) -> np.ndarray:
        # UDP can occasionally deliver late packets from prior commands
        # (e.g. set_normalized_target). Retry a few times until we get a
        # valid state payload.
        last_error: str | None = None
        for _ in range(3):
            response = self._udp_request({"cmd": "get_state", "ts": time.time()}, expect_reply=True)
            if not response:
                last_error = "No response"
                continue
            if not response.get("ok", False):
                last_error = str(response.get("error", "ok=false"))
                continue

            state = response.get("state", None)
            if state is None:
                last_error = f"State missing in response: keys={list(response.keys())}"
                continue

            state_arr = np.asarray(state, dtype=float).reshape(-1)
            if state_arr.size != 2:
                last_error = f"Invalid state shape: {state_arr.shape}"
                continue
            return state_arr

        raise RuntimeError(f"Failed to get remote state ({last_error})")

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        self.close()
