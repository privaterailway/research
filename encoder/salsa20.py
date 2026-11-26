from . import Encoder
from Crypto.Cipher import Salsa20


class Salsa20Encoder(Encoder):
    """Salsa20 stream cipher, 256-bit key, 64-bit nonce (derived from the shared IV)."""

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        if self.cipher_obj is None:
            self.cipher_obj = Salsa20.new(key=self.key, nonce=self.iv)

        return self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:8]
        self.cipher_obj = None
