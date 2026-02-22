# verify_setup.sh
#!/bin/bash

echo "================================"
echo "FRA361 Environment Verification"
echo "================================"

# Check Python
echo -n "Python version: "
python3 --version

# Check MuJoCo
echo -n "MuJoCo: "
python3 -c "import mujoco; print('✅ Version', mujoco.__version__)" 2>/dev/null || echo "❌ Not installed"

# Check Gymnasium
echo -n "Gymnasium: "
python3 -c "import gymnasium; print('✅ Version', gymnasium.__version__)" 2>/dev/null || echo "❌ Not installed"

# Check PyTorch
echo -n "PyTorch: "
python3 -c "import torch; print('✅ Version', torch.__version__)" 2>/dev/null || echo "❌ Not installed"

# Check CUDA
echo -n "CUDA: "
python3 -c "import torch; print('✅ Available' if torch.cuda.is_available() else '❌ Not available')" 2>/dev/null

# Check filterpy
echo -n "Filterpy: "
python3 -c "import filterpy; print('✅ Installed')" 2>/dev/null || echo "❌ Not installed"

# Check scikit-learn
echo -n "Scikit-learn: "
python3 -c "import sklearn; print('✅ Version', sklearn.__version__)" 2>/dev/null || echo "❌ Not installed"

echo "================================"
echo "Verification complete!"
echo "================================"