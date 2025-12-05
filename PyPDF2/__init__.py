class PdfReader:
    """Minimal PDF reader counting pages.

    This is a lightweight substitute for the external PyPDF2 package,
    sufficient for counting form-feed separated pages produced by the
    fallback PDF exporter.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        # Pages are separated by form-feed characters or /Type /Page markers
        if b"\f" in self.data:
            count = self.data.count(b"\f") + 1
        else:
            count = self.data.count(b"/Type /Page")
        self.pages = [object()] * count
