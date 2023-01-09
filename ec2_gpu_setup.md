

## Update default python to python3 

`$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1`

## Running Jupyter as a public server

To do this, you will need a jupyter config file. You can generate the config file using the following:

`$ jupyter notebook --generate-config`

This will generate a jupyter_notebook_config.py in the directory ` /home/ubuntu/.jupyter/`

### Listen on all addresses 

Uncomment the following lines and change them to the ones shown in the jupyter_notebook_config.py:

```c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
````

### Generate a password 
Instead of using the security token (which is a long string), configure the Jupyter notebook server to use password instead:

```
$ jupyter notebook password
Enter password:  ****
Verify password: ****
[NotebookPasswordApp] Wrote hashed password to /home/ubuntu/.jupyter/jupyter_notebook_config.json
```

### Set up SSL (Optional)

You only need to configure this if you are not using nginx as a reverse proxy. 

**Note: for ipywebrtc or any camera activation, the web access must be on secured channel.**

`$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nbserverkey.key -out nbservercert.pem`

copy the private key and the certficate to a secure directory e.g.  `~/.ssl`

Now uncomment and change the following to point to the cert and key files and also change the port to 443 to ensure it can be accessed from behind the firewall:

`c.NotebookApp.certfile = u'/home/ubuntu/.ssl/nbservercert.pem'`

`c.NotebookApp.keyfile = u'/home/ubuntu/.ssl/nbserverkey.key'`

`c.NotebookApp.port = 443`

`c.NotebookApp.allow_remote_access = True`

You need to be root to start the notebook server since port 443 is privileged port:

`sudo notebook server --allowed-root`


### Install ipywebrtc

`$ sudo pip install ipywebrtc`


## Setup NGINX as reverse proxy for Jupyter notebook

### Install nginx

`$ sudo apt-get update`

`$ sudo apt-get install nginx`

### Change the base url 

In jupyter config, uncomment and edit the following line: 

`c.NotebookApp.base_url = '/notebook'`

### Edit the nginx.conf file to point to the notebook server

Edit the configuration file at /etc/nginx/site-enabled/default. 
Add the following block inside the server block: 

```
location /notebook {
    proxy_pass http://localhost:8080;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header Host $http_host;
    proxy_http_version 1.1;
    proxy_redirect off;
    proxy_buffering off;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
}
```
### configure SSL for NGINX 

Generate the cert file and key file:

`sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/nginx-selfsigned.key -out /etc/ssl/certs/nginx-selfsigned.crt`

Create a configuration snippet file:
`sudo nano /etc/nginx/snippets/self-signed.conf` a

and insert the following in the file:

```
ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;
ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;
```
Next, create a second configuration snippet that points to our newly-generated SSL key and certificate. To do this, issue the command:

`sudo nano /etc/nginx/snippets/ssl-params.conf`

In that new file, add the following contents:

```
ssl_protocols TLSv1.2;
ssl_prefer_server_ciphers on;
ssl_dhparam /etc/ssl/certs/dhparam.pem;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_ecdh_curve secp384r1; # Requires nginx >= 1.1.0
ssl_session_timeout  10m;
ssl_session_cache shared:SSL:10m;
ssl_session_tickets off; # Requires nginx >= 1.5.9
# ssl_stapling on; # Requires nginx >= 1.3.7
# ssl_stapling_verify on; # Requires nginx => 1.3.7
resolver 8.8.8.8 8.8.4.4 valid=300s;
resolver_timeout 5s;
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
```

We also need to generate the dhparam.pem file with the command:

`sudo openssl dhparam -out /etc/ssl/certs/dhparam.pem 2048`

The above command will take some time.

Next we need to edit the /etc/nginx/site-enabled/default. Edit the server block (which contains the notebook block above): 

```
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    include snippets/self-signed.conf;
    include snippets/ssl-params.conf;

    server_name _;

    root /var/www/example.com/html;
    index index.html index.htm index.nginx-debian.html;
    
    server_name $hostname 
    
    location / {
                # First attempt to serve request as file, then
                # as directory, then fall back to displaying a 404.
        #try_files $uri $uri/ =404;
        proxy_pass http://localhost:8080;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Host $http_host;
        proxy_http_version 1.1;
        proxy_redirect off;
        proxy_buffering off;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    location /tensorboard   {
        return 302 /tensorboard/;
    }

    location /tensorboard/ {
        proxy_pass http://localhost:6006/;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Host $http_host;
        proxy_http_version 1.1;
        proxy_redirect off;
        proxy_buffering off;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
     }
```

** Note it is important to specify http://localhost:6006/ so that the context will be stripped off

Below that add a new server block (to perform an HTTPS redirect) like so:

```
server {
    listen 80;
    listen [::]:80;

    server_name _;

    return 302 https://$server_name$request_uri;
}
```
Note that the `default` in `sites-enabled` is a symbolic link to `sites-available/default`

Perform a check using:

`sudo ufw app list`

You should be able to see that HTTPS is listed. 

Restart NGINX with with the command:

`sudo systemctl restart nginx`

### Running Jupyter notebook as a service

Create the following file jupyter.service in the directory /lib/systemd/system/jupyter.service: 

```
[Unit]
Description=Jupyter Notebook

[Service]
Type=simple
PIDFile=/run/jupyter.pid
ExecStart=/usr/local/bin/jupyter-notebook --config=/home/ubuntu/.jupyter/jupyter_notebook_config.py
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu
Restart=always
RestartSec=10
#KillMode=mixed

[Install]
WantedBy=multi-user.target

```

Then enable the service by:

`sudo systemctl enable jupyter.service`

And then reload the daemon and run the service:

`sudo systemctl daemon-reload`

`sudo systemctl restart jupyter.service`

### Install ipython kernel to notebook

`conda activate tf2env`

`conda install pip`

`conda install ipykernel`

`python -m ipykernel install --user --name tf2env --display-name "Python (tf2env)"`



### Notes for GPU 

Somehow when the jupyter notebook was started using the systemctl, the LD_LIBRARY_PATH does not include the cuda libraries, so it will fail.  Need to include an ENVIRONMENT line in jupyter.service file: 

```
[Unit]
Description=Jupyter Notebook

[Service]
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
Type=simple
...

```

### Notes for Tensorflow 2.0.0-rc1

We need to update the default cuda that come with the AWS Deep Learning AMI. The version installed is 7.4, but the tensorflow 2.0.0 rc1 was compiled with cudnn 7.6.  Download the amd64 version of 7.6 from: 

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.3.30/Production/10.0_20190822/Ubuntu18_04-x64/libcudnn7_7.6.0.30-1%2Bcuda10.0_amd64.deb. Note you need the exact version and use the one for cuda 10.0. 


and use the following command to install it: 

``sudo dpkg -i libcudnn7_7.6.0.30-1+cuda10.0_amd64.deb``

This will install the .so files inside the /usr/local/x86_64-linux-gnu directory. 

rm the old one: 

``rm -rf /usr/local/cuda/lib64/libcudnn.so.17.4``

and relink to the new ones: 

``ln -sfn /usr/local/x86_64-linux-gnu/libcudnn.so.7.6.0``


## UNLOAD NVIDIA DRIVER

When you encountered "NVML: Driver/library version mismatch": you can do the following:

```
lsmod | grep nvidia

sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia_uvm

sudo rmmod nvidia

```

if you get an error like rmmod: ERROR: Module nvidia is in use, then:

```

sudo lsof /dev/nvidia*

sudo rmmod nvidia_modeset (if it is used by nvidia-modeset)

```

and then do the `sudo rmmod nvidia` again.





