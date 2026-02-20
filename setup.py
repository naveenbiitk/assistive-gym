from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistive_gym'))

# with open("README.md", "r") as f:
#     long_description = f.read()

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive_gym', 'envs', 'assets')
data_files = [os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assistive_gym', 'config.ini')]

for root, dirs, files in os.walk(directory):
    for fn in files:
        data_files.append(os.path.join(root, fn))

setup(name='assistive-gym',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        # ---------- core runtime (needed by assistive_gym at import time) ----------
        'gym>=0.25,<=0.26.2',       # env API; <=0.25.2 keeps Colab's dopamine-rl happy
        'pybullet>=3.2.5',          # physics sim
        'numpy>=1.22,<2.1',         # Colab's numba needs <2.1
        'screeninfo==0.6.1',        # monitor detection (base_env, env)
        'keras>=3.0',               # loads arm-limits model (human_creation)
        'ray[rllib]>=2.9',          # all *_envs.py import ray at top level
        'scipy>=1.9',               # human.py spatial transforms
        'numpngw>=0.1.4',           # learn.py animated-png writer
        # ---------- indirect / backend for keras ----------
        # tensorflow is NOT imported directly; keras needs a backend.
        # On Colab TF 2.19.x is pre-installed — do NOT force 2.20.
        'tensorflow>=2.16,<2.21',
    ],
    # description='Physics simulation for assistive robotics and human-robot interaction.',
    # long_description=long_description,
    # mention extra lib installed;
    long_description_content_type="text/markdown",
    url='https://github.com/Healthcare-Robotics/assistive-gym',
    author='Zackory Erickson',
    author_email='zackory@gatech.edu',
    license='MIT',
    platforms='any',
    keywords=['robotics', 'assitive robotics', 'human-robot interaction', 'physics simulation'],
    package_data={'assistive_gym': data_files},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows', 'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS', 'Intended Audience :: Science/Research',
        "Programming Language :: Python",
        'Programming Language :: Python :: 3.6', 'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Robot Framework'
    ],
)
