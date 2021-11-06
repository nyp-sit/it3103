#!/bin/sh
set -x
/ngrok start --config ~/.ngrok2/ngrok.yml --config /drive/ngrok_stuff/ngrok_config.yml --log=stdout ssh tensorboard
