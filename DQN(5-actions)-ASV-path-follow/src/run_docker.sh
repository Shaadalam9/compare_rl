docker run -it --rm --gpus all --name dqn_asv \
    --net=host \
    --env="DISPLAY" \
    --workdir="/home/docker/DQN(5-actions)-ASV-path-follow/src" \
    --volume=$(pwd):"/home/docker/DQN(5-actions)-ASV-path-follow/src" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    abhilashiit/dqn_asv:2.0 bash
