# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:56:33 2022

@author: bjh3
"""

from setuptools import setup, find_packages

setup(
    name = 'tomosans',
    version = '1.0',
    author = 'Benjamin Heacock',
    author_email = 'benjamin.heacock@gmail.com',
    description = 'A package for simulating spin textures from SANS data',
    packages = ['tomosans'],
    install_requires=['numpy','scipy','pyfftw','tqdm','ffmpeg'],
    package_data={"":["README.MD","TomoSANS/MagEmKey.png"]}
    )
