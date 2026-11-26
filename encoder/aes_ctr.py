from . import Encoder
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class AesCtrEncoder(Encoder):
    """AES-256 in CTR mode (stream-like behaviour, no padding requirements)."""

    def __init__(self):
        self.backend = default_backend()
        self.encryptor = None

    def encrypt(self, text: bytes) -> bytes:
        if self.encryptor is None:
            cipher = Cipher(algorithms.AES(self.key), modes.CTR(self.iv), backend=self.backend)
            self.encryptor = cipher.encryptor()

        return self.encryptor.update(text) + self.encryptor.finalize()

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.encryptor = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:16]
        self.encryptor = None
