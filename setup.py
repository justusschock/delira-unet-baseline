from setuptools import setup, find_packages
import os
import versioneer


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements = resolve_requirements(os.path.join(os.path.dirname(__file__),
                                                 'requirements.txt'))

readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))


setup(
    name='unet',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/justusschock/delira-unet-baseline',
    test_suite="unittest",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.5",
    author="Justus Schock",
    author_email="justus.schock@rwth-aachen.de",
    license="BSD-2",
    packages=find_packages()
)
