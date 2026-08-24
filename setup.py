"""ZeypherLive — Setup Script"""
from setuptools import setup, find_packages

setup(
    name="zeypherlive",
    version="1.0.0",
    description="Real-Time Body & Face Swap Engine with Voice Changer, OBS Bridge, and Mobile Support",
    author="ZeypherLive",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "mediapipe>=0.10.0",
        "PyQt5>=5.15.0",
        "pyaudio>=0.2.13",
        "scipy>=1.11.0",
        "aiohttp>=3.8.0",
        "websockets>=12.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "obs": ["pyvirtualcam>=0.14.0"],
        "onnx": ["onnxruntime>=1.16.0"],
        "android": ["kivy>=2.2.0", "buildozer>=1.5.0"],
        "lucy": ["aiohttp>=3.8.0", "websockets>=12.0"],
    },
    entry_points={
        "console_scripts": [
            "zeypher-desktop=run_desktop:main",
            "zeypher-cli=run_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
    ],
)
