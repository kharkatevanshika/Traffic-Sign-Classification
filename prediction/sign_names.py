import os
import pandas as pd


# ============================================================
# TRAFFIC SIGN LABEL LOADER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def find_label_csv():
    """
    Search the complete project for label.csv.
    """

    print("\nSearching for label.csv...")

    found_files = []

    for root, dirs, files in os.walk(BASE_DIR):

        for file in files:

            if file.lower() == "labels.csv":

                found_files.append(
                    os.path.join(root, file)
                )


    if len(found_files) == 0:

        print("\nERROR: label.csv was not found!")

        print(
            "\nSearch location:"
        )

        print(BASE_DIR)

        return None


    print(
        f"\nFound {len(found_files)} labels.csv file(s)."
    )

    for file in found_files:

        print(
            "✓",
            file
        )


    # Use the first label.csv found
    return found_files[0]


# ============================================================
# FIND CSV
# ============================================================

LABEL_FILE = find_label_csv()


# ============================================================
# LOAD SIGN NAMES
# ============================================================

def load_sign_names():

    if LABEL_FILE is None:

        return {}


    try:

        print(
            "\nLoading label file:"
        )

        print(
            LABEL_FILE
        )


        df = pd.read_csv(
            LABEL_FILE
        )


    except Exception as e:

        print(
            "\nERROR: Could not read label.csv"
        )

        print(e)

        return {}


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    print(
        "\nCSV columns:"
    )

    print(
        df.columns.tolist()
    )


    # ========================================================
    # FIND CLASS ID COLUMN
    # ========================================================

    id_column = None


    possible_id_columns = [

        "ClassId",
        "ClassID",
        "class_id",
        "classid",
        "Class Id",
        "class id",
        "ID",
        "Id",
        "id"

    ]


    for column in possible_id_columns:

        if column in df.columns:

            id_column = column

            break


    # ========================================================
    # FIND SIGN NAME COLUMN
    # ========================================================

    name_column = None


    possible_name_columns = [

        "Name",
        "name",
        "SignName",
        "Sign Name",
        "sign_name",
        "sign name",
        "Label",
        "label",
        "Description",
        "description"

    ]


    for column in possible_name_columns:

        if column in df.columns:

            name_column = column

            break


    # ========================================================
    # IF STANDARD COLUMN NAMES NOT FOUND
    # ========================================================

    if id_column is None or name_column is None:

        print(
            "\nCould not automatically identify "
            "the required columns."
        )

        print(
            "\nAvailable columns:"
        )

        for column in df.columns:

            print(
                " -",
                column
            )


        # ----------------------------------------------------
        # Try first two columns automatically
        # ----------------------------------------------------

        if len(df.columns) >= 2:

            print(
                "\nTrying first two columns..."
            )

            id_column = df.columns[0]

            name_column = df.columns[1]


        else:

            print(
                "\nERROR: CSV does not contain "
                "enough columns."
            )

            return {}


    # ========================================================
    # DISPLAY SELECTED COLUMNS
    # ========================================================

    print(
        "\nClass ID column:"
    )

    print(
        id_column
    )


    print(
        "\nSign Name column:"
    )

    print(
        name_column
    )


    # ========================================================
    # CREATE DICTIONARY
    # ========================================================

    sign_names = {}


    for _, row in df.iterrows():

        try:

            class_id = int(
                row[id_column]
            )


            sign_name = str(
                row[name_column]
            ).strip()


            if sign_name:

                sign_names[class_id] = sign_name


        except Exception:

            continue


    # ========================================================
    # RESULT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "SIGN LABELS LOADED"
    )

    print(
        "=" * 60
    )


    print(
        "Total sign names:",
        len(sign_names)
    )


    print(
        "\nClass -> Sign Name"
    )

    print(
        "-" * 60
    )


    for class_id in sorted(sign_names):

        print(
            f"{class_id:>3} -> "
            f"{sign_names[class_id]}"
        )


    return sign_names


# ============================================================
# LOAD LABELS
# ============================================================

SIGN_NAMES = load_sign_names()


# ============================================================
# GET SIGN NAME
# ============================================================

def get_sign_name(class_id):

    try:

        class_id = int(
            class_id
        )

    except Exception:

        return "Unknown Traffic Sign"


    return SIGN_NAMES.get(
        class_id,
        f"Unknown Traffic Sign (Class {class_id})"
    )