from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class TripleDesEncoder(Encoder):
    def __init__(self):
        self.backend = default_backend()
        self.encryptor = None

    def encrypt(self, text: bytes) -> bytes:
        assert len(text) % 8 == 0, "Text length must be a multiple of 8 bytes for 3DES"

        if self.encryptor is None:
            cipher = Cipher(algorithms.TripleDES(self.key), modes.CBC(self.iv), backend=self.backend)
            self.encryptor = cipher.encryptor()

        ciphertext = self.encryptor.update(text) + self.encryptor.finalize()

        return ciphertext

    def generate_key(self, key: bytes):
        self.key = key[:24]
        self.encryptor = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:8]
        self.encryptor = None
