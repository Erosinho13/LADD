#!/usr/bin/env bash

########################################################################################################################

function init() {
  device_ids=-1
  batch_size=-1
  framework=-1
  fw_task=-1
}

function run() {
	cd "${path}"/src/ || exit 1
	${command}
}

########################################################################################################################

path=$(pwd)
port=$(python get_free_port.py)
config_path="$1"
command="python3 -W ignore -m torch.distributed.launch --nproc_per_node * --master_port=${port} run.py"

while IFS="=" read -r arg value; do

  if [ "${arg}" != "" ]; then
    if [ "${value}" = "" ]; then
      command="${command} --${arg}"
    else
      declare "${arg}"="${value}"
      if [ "${arg}" = "device_ids" ]; then
        device_ids="${value}"
        device_ids="${device_ids:1:-1}"
        IFS=' ' read -r -a device_ids_array <<< "${device_ids}"
        num_devices=${#device_ids_array[@]}
        command="${command} --device_ids ${device_ids}"
      elif [ "${arg}" = "batch_size" ]; then
        batch_size_per_device=$((batch_size/num_devices))
        command="${command} --batch_size ${batch_size_per_device%.*}"
      else
        command="${command} --${arg} ${value}"
      fi
    fi
  fi

done < "$config_path"

batch_size_per_device=$((batch_size/num_devices))
command=${command//[*]/${num_devices}}

########################################################################################################################

echo "GPUs in usage:" "${device_ids_array[@]}"
echo "Running ${framework} ${fw_task} experiment..."
run
echo "Done."
