# Transfer Learning

This project is a simple implementation of a transformer model for image captioning.

## Setup

```bash
conda activate mlx-4
```

on a new machine, you can import the environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
```

## Connecting to Computa

```bash
# jump into the ssh folder
cd ~/.ssh
# create a new ssh key
echo "<PK>" > q3km682zd7ush1
# change the file permissions so other actors cannot read your pk
chmod 600 q3km682zd7ush1
# connect to the server at this ip at this port using the ssh key
ssh root@209.53.88.242 -p 13511 -i q3km682zd7ush1
# remove the ssh key when you are done
rm q3km682zd7ush1
# exit when done
exit
```