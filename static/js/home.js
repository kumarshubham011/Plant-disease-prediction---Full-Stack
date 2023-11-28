function changedata(){
    change_var = document.getElementById("#data-change-helper");
    counter = 1
    change_var.innerHTML = `<div style="height: 27rem;" class="card">
            <div style="display: flex;">
                <span class="head-style">
                    {% for key in data_value[${counter}].keys() %}
                        <h2 style="font-size: 3rem; padding-top: 5px; color: white; position: relative; right: 3rem;">{{key}}</h2>
                        <div style="height:6rem; width: 3rem; background-color: #f8f3e8; position: absolute; right: 11rem; border-radius: 50% 0 0 50%;"></div>
                </span>
                <img style="border-radius: 50%; height: 10rem; width: 12rem; position: absolute; right: 1rem;" src={{disease_image_data[key]}} />
            </div>
            <div class="card-disease-data">
                {% for list_data in data_value[${counter}].values() %}
                    {% for bullets in list_data %}
                        <i class="bi bi-arrow-right"></i>&nbsp;{{bullets}}<br><br>
                    {%endfor%}
                {%endfor%}
            </div>
            {%endfor%}
        </div>
    `
}