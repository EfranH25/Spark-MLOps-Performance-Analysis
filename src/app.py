from services import utils

if __name__ == "__main__":
    print("starting")

    utils.get_data("creditcardfraud")
    utils.cleanup_data("creditcardfraud", clean_all=False)
    print("done")
