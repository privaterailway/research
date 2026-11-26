from . import Encoder
from Crypto.Cipher import CAST


class Cast5Encoder(Encoder):
    """CAST5 (CAST-128), 128-bit key, CBC mode, 64-bit block."""

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        assert len(text) % 8 == 0, "Text length must be a multiple of 8 bytes for CAST5"

        if self.cipher_obj is None:
            self.cipher_obj = CAST.new(self.key, CAST.MODE_CBC, iv=self.iv)

        return self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:16]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:8]
        self.cipher_obj = None
