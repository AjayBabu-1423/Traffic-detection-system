from django.apps import AppConfig


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'APP'
    
class PasswordgeneratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'passwordGenerator'