***How to navigate through the application***

1.Open the GitHub repository.

2.Click on "Code" and select "Open with Codespaces".

3.Wait for the Codespace environment to initialize.

4.Once ready, open the terminal and run:

***Start the docker containers***
```bash
docker compose up -d
```

***Check what containers are running***
```bash
docker ps
```
Copy the id of postgres container

***Check the ip of postgres***
```bash
docker inspect id
```

***Navigate to the flour-classification-app***
```bash
cd flour-classification-app
```

After getting to the flour-classification-app,navigate to the routes.py and replace the DATABASE_URI host with the ip address you copied above.

***Run the application***
```bash
python run.py
```
Follow the port fowarded and open in the browser

***Login in to the application***
login with username and password ,they are all in the users.json

Voila,go ahead to use the application