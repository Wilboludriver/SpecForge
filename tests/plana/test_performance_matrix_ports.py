"""Real local TCP states must satisfy the same raw-bind gate as the launcher."""
from pathlib import Path
import socket
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/plana'))
from run_performance_matrix import port_snapshot


class RawPortTests(unittest.TestCase):
    def test_bound_nonlistener_is_busy_and_can_be_used_after_owner_closes(self):
        with socket.socket() as owner:
            owner.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            owner.bind(('127.0.0.1',0));port=owner.getsockname()[1]
            with socket.socket() as old_probe:
                old_probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                old_probe.bind(('0.0.0.0',port))
            self.assertEqual([row['port'] for row in port_snapshot([port])],[port])
        self.assertEqual(port_snapshot([port]),[])

    def test_time_wait_is_busy_even_when_reuseaddr_would_allow_binding(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
            listener.listen(1)
            with socket.socket() as client:
                client.connect(('127.0.0.1',port))
                with listener.accept()[0] as accepted:
                    # Make the server endpoint the active closer / TIME_WAIT owner.
                    accepted.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1),b'')
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(accepted.recv(1),b'')
        with socket.socket() as old_probe:
            old_probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            old_probe.bind(('0.0.0.0',port))
        self.assertEqual([row['port'] for row in port_snapshot([port])],[port])


if __name__=='__main__':unittest.main()
