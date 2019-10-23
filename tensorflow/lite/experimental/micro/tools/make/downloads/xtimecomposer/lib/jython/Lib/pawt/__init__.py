import sys
from java import awt

def test(panel, size=None, name='AWT Tester'):
    f = awt.Frame(name, windowClosing=lambda event: sys.exit(0))
    if hasattr(panel, 'init'):
        panel.init()

    f.add('Center', panel)
    f.pack()
    if size is not None:
        f.setSize(apply(awt.Dimension, size))
    f.setVisible(1)
    return f

class GridBag:
    def __init__(self, frame, **defaults):
        self.frame = frame
        self.gridbag = awt.GridBagLayout()
        self.defaults = defaults
        frame.setLayout(self.gridbag)

    def addRow(self, widget, **kw):
        kw['gridwidth'] = 'REMAINDER'
        apply(self.add, (widget, ), kw)

    def add(self, widget, **kw):
        constraints = awt.GridBagConstraints()

        for key, value in self.defaults.items()+kw.items():
            if isinstance(value, type('')):
                value = getattr(awt.GridBagConstraints, value)
            setattr(constraints, key, value)
        self.gridbag.setConstraints(widget, constraints)
        self.frame.add(widget)
