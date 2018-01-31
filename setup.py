from setuptools import setup
from app import db

setup(
    name='game-face',
    packages=['game-face'],
    include_package_data=True,
    install_requires=[
        'flask',
        'flask-sqlalchemy',
        'plotly'
    ],
)