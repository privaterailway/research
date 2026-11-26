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
        # XChaCha20 needs a 24-byte nonce; datagen supplies a 32-byte random IV.
        assert len(iv) >= 24, "XChaCha20 needs an IV of at least 24 bytes"
        self.iv = iv[:24]
        self.cipher_obj = None
