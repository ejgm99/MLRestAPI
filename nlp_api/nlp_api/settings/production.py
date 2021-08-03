from nlp_api.settings.common import *
import os
DEBUG = False

import secrets

length = 50
chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'

SECRET_KEY = ''.join(secrets.choice(chars) for i in range(length))


# SECURITY WARNING: update this when you have the production host
ALLOWED_HOSTS = ['127.0.0.1', 'localhost','http://ec2-3-136-18-160.us-east-2.compute.amazonaws.com/']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'postgres',
        'USER': 'postgres',
        'PASSWORD': 'p88t90k95e99',
        'HOST': 'general.cqx4pjfopx41.us-east-2.rds.amazonaws.com',
        'PORT': '5432',
    }
}
