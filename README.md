## BiometricSheduleControl backend 

First of all, you need to install python 3.6.5 and then install the requirements inside the file with the same name.

```bash
pip install -r requirements.txt
```

After cloning or downloading this branch, you need to generate the database and tables, sqlite by default. So you need to run inside the folder the command below.

```bash
python manage.py makemigrations 
```
and then. 
```bash
python manage.py migrate
```

After that, you need to insert the data, inside the exports folder, in the database. Profile, state and type are the mandatory inserts.
Then modify the settings.py file inside the bsc folder.

Now, you can run the code, example.
```bash
python manage.py 0.0.0.0:80
```
The runserver.py is to run the code in production, with a nginx server and waitress.

## Warning 

This app started as academic work and continued as a project in a university Chair award (Cátedra MN, University of Vigo). So this app is in Spanish.
