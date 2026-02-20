from setuptools import setup, find_packages

setup(
    name="motion_skill",  # You can name this anything
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "hydra-core",
        "wandb",
        # Add other dependencies here
    ],
)
