from . import Encoder
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class BlowfishEncoder(Encoder):
    def __init__(self):
        self.backend = default_backend()
        self.encryptor = None

    def encrypt(self, text: bytes) -> bytes:
        assert len(self.key) == 32, "Key length must be 32 bytes for Blowfish"

        if self.encryptor is None:
            cipher = Cipher(algorithms.Blowfish(self.key), modes.CBC(self.iv), backend=self.backend)
            self.encryptor = cipher.encryptor()

        return self.encryptor.update(text) + self.encryptor.finalize()

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.encryptor = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:8]
        self.encryptor = None
