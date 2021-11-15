terraform {
  required_providers {
    xpd = {
      source = "0x2b3bfa0/xpd",
    }
  }
}

provider "xpd" {}

variable "name" {}
variable "repo_token" {}

resource "xpd_task" "task" {
  cloud     = "aws"
  name      = var.name
  directory = "."

  environment = {
    REPO_TOKEN = var.repo_token
  }

  script = <<-END
    #!/bin/bash
    export HOME=/root

    curl --silent --location https://deb.nodesource.com/setup_14.x | bash
    apt update && apt install --yes build-essential python3-pip git nodejs
    npm install --global --unsafe @dvcorg/cml

    # export EPOCHS=1
    # export S3_BUCKET=s3://daviddvctest/$NAME
    # pip3 install --requirement requirements.txt
    # python3 train.py

    date > file
    cml pr file 
  END
}
