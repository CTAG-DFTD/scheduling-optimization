import os
import numpy as np
import time

# Read environment variable ONCE (module level)
QAOA_SCRIPT_PATH = os.environ.get("MAXCUT_QAOA_SCRIPT")

if QAOA_SCRIPT_PATH is None:
    raise EnvironmentError(
        "Environment variable MAXCUT_QAOA_SCRIPT is not set"
    )

def maxcut_qaoa(maxcut_matrix, day, var_mapping, num_slots):

    exchange_path = os.path.join(os.getcwd(), "exchange.npy")
    print("\n")

    while True:
        np.save(exchange_path, maxcut_matrix)

        command = (
            f"python {QAOA_SCRIPT_PATH} "
            f"--day {day} "
            f"--path {exchange_path}"
        )

        start_time = time.time()

        return_code = os.system(command)
        maxcut_bitstring = np.load(exchange_path).tolist()

        elapsed_time = time.time() - start_time

        jobs = [
            qraise.removeprefix("qraise_")
            for qraise in os.listdir()
            if qraise.startswith("qraise_")
        ]
        [os.system(f"scancel {job}") for job in jobs]
        time.sleep(30)

        if return_code != 0:
            print(f"Script failed with return code: {return_code}")
            break

        if len(get_assigment_solution(var_mapping, maxcut_bitstring)) == num_slots:
            print(f"best bitstring day {day} = {maxcut_bitstring}")
            print(f"Assignment success day {day}")
            break

        print(f"Assignment violation day {day}")

    return maxcut_bitstring, elapsed_time
