# Config params
$image_name = "waymo-od-image"
$container_name = "waymo-od-container"
$volume_local_path = "C:/Users/chris/GitHub/waymo-challenges"
$volume_container_path = "/workspace/hostfiles"
$port = 8001

# Build docker image
docker build -t $image_name .

# Remove existing container if applicable
$existing_container = docker ps -a --filter "name=$container_name" --format "{{.ID}}"
if ($existing_container) {
    docker stop $existing_container
	docker rm $existing_container
	Write-Output "Remove existing container: $existing_container"
} else {
	Write-Output "No existing container found"
}

# Run docker container
docker run -v ${volume_local_path}:${volume_container_path} -it --name $container_name -p ${port}:${port} $image_name
#docker run -it --name $container_name -p ${port}:${port} $image_name
