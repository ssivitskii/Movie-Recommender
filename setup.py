from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="movie-recommender",
    version="1.0.0",
    author="Stepan Sivitskii",
    author_email="stepan.sivitsky@yandex.ru",
    description="A movie recommendation system using collaborative filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ssivitskii/Movie-Recommender",
    project_urls={
        "Bug Tracker": "https://github.com/ssivitskii/Movie-Recommender/issues",
        "Documentation": "https://github.com/ssivitskii/Movie-Recommender#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=["tests", "notebooks", "data"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.270",
            "mypy>=1.0.0",
        ],
        "web": [
            "streamlit>=1.20.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "movie-train=src.train:main",
            "movie-predict=src.predict:main",
            "movie-evaluate=src.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

