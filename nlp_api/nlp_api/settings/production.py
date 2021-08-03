from nlp_api.settings.common import *
import os
DEBUG = False

SECRET_KEY = os.environ['SECRET_KEY']

# SECURITY WARNING: update this when you have the production host
ALLOWED_HOSTS = ['127.0.0.1', 'localhost','https://nlp-env.eba-m5vuny8c.us-west-1.elasticbeanstalk.com/']
