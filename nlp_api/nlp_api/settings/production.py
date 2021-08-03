from nlp_api.settings.common import *
import os
DEBUG = False

SECRET_KEY = os.environ['SECRET_KEY']

# SECURITY WARNING: update this when you have the production host
ALLOWED_HOSTS = ['127.0.0.1', 'localhost','https://nlp-env.eba-m5vuny8c.us-west-1.elasticbeanstalk.com/']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'postgres',
        'USER': 'postgres',
        'PASSWORD': 'p88t90k95e99',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
