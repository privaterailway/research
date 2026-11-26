from . import Encoder
import hashlib
from Crypto.Cipher import ARC4


class Rc4Encoder(Encoder):
    """RC4 (ARC4) stream cipher, 256-bit key.

    RC4 has no IV, so with a plain key every message of a key group would be encrypted with the
    SAME keystream. To avoid this keystream reuse the per-message RC4 key is derived as
    SHA-256(key || iv), which is the usual way to give RC4 a per-message nonce.
    """

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        if self.cipher_obj is None:
            self.cipher_obj = ARC4.new(hashlib.sha256(self.key + self.iv).digest())

        return self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv
        self.cipher_obj = None
