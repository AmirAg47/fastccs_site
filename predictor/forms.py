from django import forms

from django_recaptcha.fields import ReCaptchaField
from django_recaptcha.widgets import ReCaptchaV2Invisible


class CSVUploadForm(forms.Form):
    file = forms.FileField()
    captcha = ReCaptchaField(widget=ReCaptchaV2Invisible, label='')

