# Generated by Django 4.2.1 on 2023-05-20 17:11

from django.db import migrations, models


class Migration(migrations.Migration):



    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('First_Name', models.CharField(max_length=100)),
                ('Area', models.CharField(max_length=50)),
                ('Email', models.EmailField(max_length=254)),
                ('password', models.CharField(default='pass', max_length=20)),
            ],
        ),
    ]
