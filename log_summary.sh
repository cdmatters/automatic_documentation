#!/bin/bash

function log_summary(){
    
  blink=$(tput setaf blink)
  green=$(tput setaf 2)
  black=$(tput setaf 0)
  yellow=$(tput setaf 3)
  blue=$(tput setaf 6)
  bold=$(tput bold)
  reset=$(tput sgr0)


  if [ $1 = "-f" ]; then
    shift 1
    func="tail -f"
    echo $@
  else
    func="cat"
  fi

  $func $@ | \
    sed -E -e "s/^[0-9]{4}_([0-9]{2}:[0-9]{2})\s+(INFO|DEBUG|WARNING|ERROR) -/$bold$black\1 -$reset/" -e \
              "s/ARGN:(.*)<END>/ARGN:$bold\1$reset<END>/" -e \
              "s/(ARGN:|DESC:|INFR:)/$bold$green\1$reset/" -e \
              "s/(MINIBATCHES:|TRAIN_BLEU:|VALID_BLEU:|TEST_BLEU:)/$bold\1$reset/" -e \
              "s/(----*)/$bold$black\1$reset/" -e \
              "s/(TRAINING:.*)/$bold$blue\1$reset/" -e \
              "s/(TEST:.*)/$bold$yellow\1$reset/" 
}

log_summary $@
