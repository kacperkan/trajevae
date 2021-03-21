# TrajeVAE - Controllable Human Motion Prediction from Trajectories

The official implentation of the paper "TrajeVAE - Controllable Human Motion
Prediction from Trajectories"

## Prerequisities
- [Anaconda](https://anaconda.org/) >= 4.9.2

## Data

Follow the instructions
[here](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)
on how to obtain the dataset (section `Human3.6M`). Put the resulting
`data_3d_h36m.npz` file in `data/processed/human36m-3d/`.

## Installation


### Environment
```bash
$ conda env create -f env.yml # installing the dependencies
$ conda activate trajevae # activating the environment
```

### Renderer (optional)

If you wish to render few animations or plain frames from the animation, then
you need to install a Docker image of the [mitsuba render](https://github.com/mitsuba-renderer/mitsuba)

```bash
$ cd mitsuba
$ docker build -t mitsuba:1.0 .
$ docker run -p 8000:8000 mitsuba:1.0   
```

This will run the mitsuba service, available on the port `8000`.


## Usage

We attach the pretrained model of TrajeVAE.

### Main experiments

To reproduce TrajeVAE and the baselines:
```bash
$ dvc repro
```

The commands used during the reproduction can be found in the `dvc.yaml` file.

### The experiment with the same pose but different trajectories
The experiment can be run by uncommenting lines `153-155` in the file
`trajevae/evaluators/experiment.py`. Then, run either `dvc repro` or the
following command to obtain necessary results:
```
$ python -m trajegan.evaluators.experiment data/processed/human36m-3d --config_name trajevae.h36m --module_name trajevae.TrajeVAE  --without_visualization
```

Take a look to `dvc.yaml` to see how to change to arguments to obtain results
for baselines.
 

### Rendering

The rendering requires that you have the aforementioned rendering docker
container installed. 


#### **Teaser animation**
```bash
python -m trajevae.evaluators.render_whole_scene data/processed/human36m-3d/ --config_name base_h36m --module_name trajevae.TrajeVAE  --joint_indices_to_use 3 6 13 16 --renderer_port 8000
```

#### **Adding more trajectories / Generalization**
```bash
python -m trajevae.evaluators.render_several_frames_sample_vs_traj data/processed/human36m-3d/ --config_name trajevae.h36m --module_name trajevae.TrajeVAE --renderer_port 8000 
```

#### **End pose for 10 sampled sequences**
```bash
python -m trajevae.evaluators.render_several_frames_sample_vs_traj data/processed/human36m-3d/ --config_name trajevae.h36m --module_name trajevae.TrajeVAE --renderer_port 8000
```

#### **Same pose, different trajectory**
```bash
python -m trajevae.evaluators.render_several_frames_sample_vs_different_traj data/processed/human36m-3d/ --config_name trajevae.h36m --module_name trajevae.TrajeVAE --renderer_port 8000
```

#### **Creating video clips**

```bash
$ python trajevae/utils/generate_videos.py outputs/scene-renders scene
$ python trajevae/utils/generate_videos.py outputs/renders-single-frame-sample-vs-traj single
$ python trajevae/utils/generate_videos.py outputs/renders-several-frames-sample-vs-traj several 
$ python trajevae/utils/generate_videos.py outputs/renders-several-frames-sample-vs-diff-traj several 
```

Running these commands will create corresponding video clips in the
`outputs/videos` folder.

## License
MIT