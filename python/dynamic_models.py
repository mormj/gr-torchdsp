from gnuradio import gr
import os
from jinja2 import Environment, FileSystemLoader
import subprocess
from . import load_model

class DynamicModels:
    @staticmethod
    def render_model(
        template_model_name,
        rendered_model_path,
        params_dict,
        triton_url
    ):
        templates_dir = os.path.join(gr.prefix(), 'share', 'gr-torchdsp', 'templates', template_model_name)
        params_dict['rendered_model_name'] = os.path.basename(rendered_model_path)
        # Make the model dir, rm if it already exists
        os.makedirs(rendered_model_path, exist_ok=True)
        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('config.pbtxt.j2')
        output = template.render(**params_dict)
        with open(os.path.join(rendered_model_path, 'config.pbtxt'),'w') as f:
            f.write(output)
        
        template = env.get_template('make_model.py.j2')
        output = template.render(**params_dict)
        with open(os.path.join(rendered_model_path, 'make_model.py'),'w') as f:
            f.write(output)
        
        print(f'Building torchscript model in: {rendered_model_path}')
        os.makedirs(os.path.join(rendered_model_path,'1'), exist_ok=True)
        subprocess.call(
            "python3 make_model.py",
            cwd=rendered_model_path,
            shell=True
        )

        load_model(triton_url, os.path.basename(rendered_model_path))


