module TF.Dashboard {
  /**
   * ReloadBehavior: A simple behavior for dashboards where the
   * frontendReload() function should find every child element with a
   * given tag name (e.g. "tf-chart" or "tf-image-loader")
   * and call a `reload` method on that child.
   * May later extend it so it has more sophisticated logic, e.g. reloading
   * only tags that are in view.
   */
  export function ReloadBehavior(tagName) {
    return {
      properties: {
        reloadTag: {
          type: String,
          value: tagName,
        },
      },
      frontendReload: function() {
        var elements = this.getElementsByTagName(this.reloadTag);
        Array.prototype.forEach.call(elements, function(x) { x.reload(); });
      },
    };
  }
}
