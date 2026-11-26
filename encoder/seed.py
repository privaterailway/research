from . import Encoder
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.backends import default_backend

# In newer versions of `cryptography` SEED lives in `decrepit`; older ones keep it in `algorithms`.
try:
    from cryptography.hazmat.decrepit.ciphers.algorithms import SEED
except ImportError:  # pragma: no cover
    from cryptography.hazmat.primitives.ciphers.algorithms import SEED


class SeedEncoder(Encoder):
    """SEED (KISA, South Korea), 128-bit key, CBC mode, 128-bit block."""

    def __init__(self):
        self.backend = default_backend()
        self.encryptor = None

    def encrypt(self, text: bytes) -> bytes:
        assert len(text) % 16 == 0, "Text length must be a multiple of 16 bytes for SEED"

        if self.encryptor is None:
            cipher = Cipher(SEED(self.key), modes.CBC(self.iv), backend=self.backend)
            self.encryptor = cipher.encryptor()

        return self.encryptor.update(text) + self.encryptor.finalize()

    def generate_key(self, key: bytes):
        self.key = key[:16]
        self.encryptor = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:16]
        self.encryptor = None
