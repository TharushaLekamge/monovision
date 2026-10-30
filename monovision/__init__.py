"""MonoVision: monocular 3D reconstruction in poorly visible environments.

Pipeline (Ilesinghe, Lekamge, Samarutilake, IJRC 3(1), 2024):
  ios_logger capture -> frames -> night->day GAN (monovision.enhance) -> SimpleRecon -> mesh
"""
__version__ = "0.1.0"
