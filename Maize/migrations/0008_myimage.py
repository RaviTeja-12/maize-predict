# Generated by Django 4.2.1 on 2023-05-27 19:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Maize', '0007_remove_users_area'),
    ]

    operations = [
        migrations.CreateModel(
            name='MyImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
            ],
        ),
    ]
