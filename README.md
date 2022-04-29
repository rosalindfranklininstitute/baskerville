# Baskerville cluster workflow and utilities

This repo contains an example workflow and methodology for using Baskerville cluster.

It also includes the `sgit` utility script which is implemented entirely in BASH so that it can run on the Baskerville login nodes where other languages are not available.

`sgit` allows you submit a cluster job to `sbatch` directly from a specific commit of a git repo.
```
sgit -d ~/jobs -r git@github.com:rosalindfranklininstitute/baskerville.git \
     -c d126b7962d434cdce5778cca47b048b84d5f6d3e \
     -j examples/multi-task-srun/job.sh
```

The above command will create a directory `~/jobs/baskerville/61e3f7875dcc4144f81d3c6855423363da701f91/<UTC>` and submit `examples/multi-task-srun/job.sh` to sbatch with job name `baskerville/61e3f7875dcc4144f81d3c6855423363da701f91/<UTC>`.

---

## Generate an ssh key pair to allow Baskerville to pull from Github.com

**THIS SSH KEY WILL AUTHENTICATE TO GITHUB.COM AS YOU PERSONALLY!**

**NEVER PLACE YOUR SSH KEYS IN SHARED PROJECT STORAGE ON BASKERVILLE!**

Place ssh keys within your home directory `~/.ssh/` so they are kept private to you.

Keys that authenticate to external services as YOU need to be kept private from EVERYONE, even your project collaborators.

**NEVER COMMIT YOUR SSH KEYS INTO A GIT REPO!**

### Generate the SSH key pair for Github.com on Baskerville

```
# Generate a new ssh keypair in your Baskerville home directory
ssh-keygen -f ~/.ssh/id_rsa_github

# Print the contents of the PUBLIC key (~/.ssh/id_rsa_github.pub)
cat ~/.ssh/id_rsa_github.pub

# NEVER expose the content of the PRIVATE key (~/.ssh/id_rsa_github)
```

### Update your SSH config file on Baskerville to use the key pair to login to Github.com

```
# Edit your ssh config file in your Baskerville home directory
nano ~/.ssh/config

# And ensure the config file contains an entry for github like below
Host github.com
  User git
  Hostname github.com
  PreferredAuthentications publickey
  IdentityFile ~/.ssh/id_rsa_github
  
# Save the file with Ctrl-O Ctrl-X
```

### Add the PUBLIC key to Github.com

```
# On your local machine in the web browser, go to https://github.com/settings/keys

# Click "New SSH Key"

# Name the key "Baskerville" to help keep track of which key is which

# Copy the content of the PUBLIC key (~/.ssh/id_rsa_github.pub) into the textbox
# It should look like: 
ssh-rsa AAAABB...BBCCCC= <user>@bask-<node>.cluster.baskerville.ac.uk

# NEVER expose the content of the PRIVATE key (~/.ssh/id_rsa_github)
```

### Test that Baskerville can now pull from Github.com using the ssh key pair

```
# Clone this git repo on Baskerville using the git@github.com style url
git clone git@github.com:rosalindfranklininstitute/baskerville.git /tmp/baskerville

# Make sure you use the git@github.com:<user>/<repo>.git syntax so that 
# git authenticates to Github.com using the SSH key pair

# If you clone using https://github.com/<user>/<repo>.git it will NOT use the ssh key pair!
```

### Place a copy of the `sgit` script from this repo into your home directory

```
# Copy the sgit script into your home directory on Baskerville
cp -v /tmp/baskerville/sgit ~/sgit

# Make sure the sgit script is has execution permissions
chmod +x ~/sgit
```

### Test the `sgit` command works by submitting the sample job from a commit in this repo

```
sgit -d ~/jobs -r git@github.com:rosalindfranklininstitute/baskerville.git \
     -c d126b7962d434cdce5778cca47b048b84d5f6d3e \
     -j examples/multi-task-srun/job.sh
```
