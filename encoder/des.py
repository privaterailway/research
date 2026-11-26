from . import Encoder
from Crypto.Cipher import DES


class DesEncoder(Encoder):
    """Single DES, 56-bit effective key, CBC mode, 64-bit block."""

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        assert len(text) % 8 == 0, "Text length must be a multiple of 8 bytes for DES"

        if self.cipher_obj is None:
            self.cipher_obj = DES.new(self.key, DES.MODE_CBC, iv=self.iv)

        return self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:8]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:8]
        self.cipher_obj = None
