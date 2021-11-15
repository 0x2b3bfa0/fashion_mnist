terraform {
  required_providers {
    xpd = {
      source = "0x2b3bfa0/xpd",
    }
  }
}

provider "xpd" {}

variable "name" {}
variable "done" {}
variable "repo_token" {}

resource "xpd_task" "task" {
  cloud     = "aws"
  name      = var.name
  directory = "."

  environment = {
    DONE = var.done
    REPO_TOKEN = var.repo_token,
  }

  script = <<-END
    #!/bin/bash
    export HOME=/root

    curl --silent --location https://deb.nodesource.com/setup_14.x | bash
    apt update
    apt install --yes build-essential git nodejs
    npm install --global --unsafe @dvcorg/cml

    AUTHORIZATION="$(printf "x-access-token:$REPO_TOKEN" | base64)"
    git config --unset http.https://github.com/.extraheader
    git config --add http.https://github.com/.extraheader \
      "AUTHORIZATION: basic $AUTHORIZATION"

    date > file
    cml pr file 

    echo "DONE:$DONE"
  END
}
