from . import Encoder
from Crypto.Cipher import ChaCha20


class XChaCha20Encoder(Encoder):
    """XChaCha20 stream cipher (ChaCha20 with 192-bit nonce), 256-bit key."""

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        if self.cipher_obj is None:
            self.cipher_obj = ChaCha20.new(key=self.key, nonce=self.iv)

        return self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        # The shared IV is 16 bytes, XChaCha20 needs a 24-byte nonce: stretch it deterministically.
        self.iv = (iv + iv)[:24]
        self.cipher_obj = None
