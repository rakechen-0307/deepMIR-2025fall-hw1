ENV_LIST="$1"

while read -r req; do
  uv pip install "$req"
done < "$ENV_LIST"