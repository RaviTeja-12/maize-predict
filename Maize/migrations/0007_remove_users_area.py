# Generated by Django 4.2.1 on 2023-05-20 18:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Maize', '0006_users_delete_customers'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='users',
            name='Area',
        ),
    ]