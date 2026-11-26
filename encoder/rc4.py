from . import Encoder
from Crypto.Cipher import ARC4


class Rc4Encoder(Encoder):
    """RC4 (ARC4) stream cipher, 256-bit key. RC4 has no IV, so the IV is accepted but ignored."""

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        if self.cipher_obj is None:
            self.cipher_obj = ARC4.new(self.key)

        return self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv  # not used by RC4
        self.cipher_obj = None
