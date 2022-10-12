from setuptools import find_packages, setup

setup(
    name="active_irl",
    version="0.1dev",
    description="Active exploration for inverse reinforcement learning.",
    long_description=open("README.md").read(),
    url="https://github.com/lasgroup/aceirl",
    install_requires=[
        "cvxpy",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "scs",
        "seaborn",
        "sacred"
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=True,
)
