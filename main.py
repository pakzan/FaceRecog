import FaceRecog

# Call FaceRecog.main() to obtain player info
# FaceRecog.main() will return values after all players' faces found for 2 seconds
# player_info = {'Player 1': [location(cm), angle(degree)], 'Player 2': ...}
# format: DICTIONARY = {STRING: [float, float]}
players_info = FaceRecog.main()

# FaceRecog problem will terminate after called
# Reload it to run the module again
import importlib
importlib.reload(FaceRecog)
new_players_info = FaceRecog.main()
