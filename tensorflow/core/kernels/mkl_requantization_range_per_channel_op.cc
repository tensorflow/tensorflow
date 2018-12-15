





<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://github.githubassets.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-lLo2nlsdl+bHLu6PGvC2j3wfP45RnK4wKQLiPnCDcuXfU38AiD+JCdMywnF3WbJC1jaxe3lAI6AM4uJuMFBLEw==" rel="stylesheet" href="https://github.githubassets.com/assets/frameworks-08fc49d3bd2694c870ea23d0906f3610.css" />
  <link crossorigin="anonymous" media="all" integrity="sha512-e6nGau96UN1peD2RzhQO751wvGlQxTD/9AcKmazYzW+60oOktOIpzo3+aNAytUQRLd8+Lt9tQx6eceASpOyyoQ==" rel="stylesheet" href="https://github.githubassets.com/assets/github-0524f09e371b3158fdf36c563063809a.css" />
  
  
  
  
  

  <meta name="viewport" content="width=device-width">
  
  <title>private-tensorflow/mkl_requantization_range_per_channel_op.cc at perchannel-int8-relu6 · NervanaSystems/private-tensorflow</title>
    <meta name="description" content="Contribute to NervanaSystems/private-tensorflow development by creating an account on GitHub.">
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    
    <meta property="og:image" content="https://avatars3.githubusercontent.com/u/9260121?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="NervanaSystems/private-tensorflow" /><meta property="og:url" content="https://github.com/NervanaSystems/private-tensorflow" /><meta property="og:description" content="Contribute to NervanaSystems/private-tensorflow development by creating an account on GitHub." />

  <link rel="assets" href="https://github.githubassets.com/">
  <link rel="web-socket" href="wss://live.github.com/_sockets/VjI6MzUxODY3MzYwOjE5ZjgzZmZkYTdjY2Q2NzEwMGFmMjc1MzAwMjZmYTdhZWJmM2Q0ZDcwZGE4NDNkNTNlYmRmMzM3ZjQ3ZmFiMzA=--94da3300c3738281cae2751fd8d6a37a696d6619">
  <meta name="pjax-timeout" content="1000">
  <link rel="sudo-modal" href="/sessions/sudo_modal">
  <meta name="request-id" content="525C:2F5E:2767BA8:3AE75FF:5C144B70" data-pjax-transient>


  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

      <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
    <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
    <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">

  <meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-request_id" content="525C:2F5E:2767BA8:3AE75FF:5C144B70" /><meta name="octolytics-dimension-region_edge" content="sea" /><meta name="octolytics-dimension-region_render" content="iad" /><meta name="octolytics-actor-id" content="38869685" /><meta name="octolytics-actor-login" content="nammbash" /><meta name="octolytics-actor-hash" content="936498e55f817fe5a13b60d3717b181dab57bcb92954e2adb801a0a40a77743d" />
<meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />



    <meta name="google-analytics" content="UA-3769691-2">

  <meta class="js-ga-set" name="userId" content="14a2863d04b53b160a3f26c2805f2d00" %>

<meta class="js-ga-set" name="dimension1" content="Logged In">



  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="nammbash">

      <meta name="expected-hostname" content="github.com">
    <meta name="js-proxy-site-detection-payload" content="ZDEyZTZhMjRhZjI3NWUwMjQyODAyYjQ5YWJhNWQ3MWIxODkxNGEzNzY2MTYwZTMxYTk0ZDc1NTJiMTY0OWQxZXx7InJlbW90ZV9hZGRyZXNzIjoiMTkyLjU1LjU0LjQxIiwicmVxdWVzdF9pZCI6IjUyNUM6MkY1RToyNzY3QkE4OjNBRTc1RkY6NUMxNDRCNzAiLCJ0aW1lc3RhbXAiOjE1NDQ4MzM5NTUsImhvc3QiOiJnaXRodWIuY29tIn0=">

    <meta name="enabled-features" content="DASHBOARD_V2_LAYOUT_OPT_IN,EXPLORE_DISCOVER_REPOSITORIES,UNIVERSE_BANNER,MARKETPLACE_PLAN_RESTRICTION_EDITOR,NOTIFY_ON_BLOCK,TIMELINE_COMMENT_UPDATES,RELATED_ISSUES,MARKETPLACE_INSIGHTS_V2">

  <meta name="html-safe-nonce" content="77d88914ed55f8f982a3d215eaf244cb384fd35d">

  <meta http-equiv="x-pjax-version" content="13f87a742a2a22fe92a18235bd33f3e2">
  

      <link href="https://github.com/NervanaSystems/private-tensorflow/commits/perchannel-int8-relu6.atom?token=AlEatZ1mwKZE-dlIobkaloSehiGpG9lxks66IX4iwA%3D%3D" rel="alternate" title="Recent Commits to private-tensorflow:perchannel-int8-relu6" type="application/atom+xml">

  <meta name="go-import" content="github.com/NervanaSystems/private-tensorflow git https://github.com/NervanaSystems/private-tensorflow.git">

  <meta name="octolytics-dimension-user_id" content="9260121" /><meta name="octolytics-dimension-user_login" content="NervanaSystems" /><meta name="octolytics-dimension-repository_id" content="92093330" /><meta name="octolytics-dimension-repository_nwo" content="NervanaSystems/private-tensorflow" /><meta name="octolytics-dimension-repository_public" content="false" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="92093330" /><meta name="octolytics-dimension-repository_network_root_nwo" content="NervanaSystems/private-tensorflow" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="false" />


    <link rel="canonical" href="https://github.com/NervanaSystems/private-tensorflow/blob/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://github.githubassets.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" class="js-site-favicon" href="https://github.githubassets.com/favicon.ico">

<meta name="theme-color" content="#1e2327">


  <meta name="u2f-support" content="true">

  <link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-in env-production page-blob">
    

  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" tabindex="1" class="p-3 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
    <div id="js-pjax-loader-bar" class="pjax-loader-bar"><div class="progress"></div></div>

    
    
    


        
<header class="Header  f5" role="banner">
  <div class="d-flex flex-justify-between px-3 ">
    <div class="d-flex flex-justify-between ">
      <div class="">
        <a class="header-logo-invertocat" href="https://github.com/" data-hotkey="g d" aria-label="Homepage" data-ga-click="Header, go to dashboard, icon:logo">
  <svg height="32" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>

      </div>

    </div>

    <div class="HeaderMenu d-flex flex-justify-between flex-auto">
      <nav class="d-flex" aria-label="Global">
            <div class="">
              <div class="header-search scoped-search site-scoped-search js-site-search position-relative js-jump-to"
  role="combobox"
  aria-owns="jump-to-results"
  aria-label="Search or jump to"
  aria-haspopup="listbox"
  aria-expanded="false"
>
  <div class="position-relative">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" data-scope-type="Repository" data-scope-id="92093330" data-scoped-search-url="/NervanaSystems/private-tensorflow/search" data-unscoped-search-url="/search" action="/NervanaSystems/private-tensorflow/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
      <label class="form-control header-search-wrapper header-search-wrapper-jump-to position-relative d-flex flex-justify-between flex-items-center js-chromeless-input-container">
        <input type="text"
          class="form-control header-search-input jump-to-field js-jump-to-field js-site-search-focus js-site-search-field is-clearable"
          data-hotkey="s,/"
          name="q"
          value=""
          placeholder="Search or jump to…"
          data-unscoped-placeholder="Search or jump to…"
          data-scoped-placeholder="Search or jump to…"
          autocapitalize="off"
          aria-autocomplete="list"
          aria-controls="jump-to-results"
          aria-label="Search or jump to…"
          data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations#csrf-token=7IFs8Xkt/hMr957pyF0YYWPwXCun3AXNhMVb7g/25+szuvD5To1iNrceURjXJlmwpxmD9PNnikWs4uTSIkYRiA=="
          spellcheck="false"
          autocomplete="off"
          >
          <input type="hidden" class="js-site-search-type-field" name="type" >
            <img src="https://github.githubassets.com/images/search-key-slash.svg" alt="" class="mr-2 header-search-key-slash">

            <div class="Box position-absolute overflow-hidden d-none jump-to-suggestions js-jump-to-suggestions-container">
              
<ul class="d-none js-jump-to-suggestions-template-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-suggestion" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0 0 13 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 0 0 0-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

</ul>

<ul class="d-none js-jump-to-no-results-template-container">
  <li class="d-flex flex-justify-center flex-items-center f5 d-none js-jump-to-suggestion p-2">
    <span class="text-gray">No suggested jump to results</span>
  </li>
</ul>

<ul id="jump-to-results" role="listbox" class="p-0 m-0 js-navigation-container jump-to-suggestions-results-container js-jump-to-suggestions-results-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-scoped-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0 0 13 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 0 0 0-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-global-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0 0 13 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 0 0 0-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>


    <li class="d-flex flex-justify-center flex-items-center p-0 f5 js-jump-to-suggestion">
      <img src="https://github.githubassets.com/images/spinners/octocat-spinner-128.gif" alt="Octocat Spinner Icon" class="m-2" width="28">
    </li>
</ul>

            </div>
      </label>
</form>  </div>
</div>

            </div>

          <ul class="d-flex pl-2 flex-items-center text-bold list-style-none">
            <li>
              <a class="js-selected-navigation-item HeaderNavlink px-2" data-hotkey="g p" data-ga-click="Header, click, Nav menu - item:pulls context:user" aria-label="Pull requests you created" data-selected-links="/pulls /pulls/assigned /pulls/mentioned /pulls" href="/pulls">
                Pull requests
</a>            </li>
            <li>
              <a class="js-selected-navigation-item HeaderNavlink px-2" data-hotkey="g i" data-ga-click="Header, click, Nav menu - item:issues context:user" aria-label="Issues you created" data-selected-links="/issues /issues/assigned /issues/mentioned /issues" href="/issues">
                Issues
</a>            </li>
              <li class="position-relative">
                <a class="js-selected-navigation-item HeaderNavlink px-2" data-ga-click="Header, click, Nav menu - item:marketplace context:user" data-octo-click="marketplace_click" data-octo-dimensions="location:nav_bar" data-selected-links=" /marketplace" href="/marketplace">
                   Marketplace
</a>                  
              </li>
            <li>
              <a class="js-selected-navigation-item HeaderNavlink px-2" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore" href="/explore">
                Explore
</a>            </li>
          </ul>
      </nav>

      <div class="d-flex">
        
<ul class="user-nav d-flex flex-items-center list-style-none" id="user-links">
  <li class="dropdown">
    <span class="d-inline-block  px-2">
      
    <a aria-label="You have no unread notifications" class="notification-indicator tooltipped tooltipped-s  js-socket-channel js-notification-indicator" data-hotkey="g n" data-ga-click="Header, go to notifications, icon:read" data-channel="notification-changed:38869685" href="/notifications">
        <span class="mail-status "></span>
        <svg class="octicon octicon-bell" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 12v1H0v-1l.73-.58c.77-.77.81-2.55 1.19-4.42C2.69 3.23 6 2 6 2c0-.55.45-1 1-1s1 .45 1 1c0 0 3.39 1.23 4.16 5 .38 1.88.42 3.66 1.19 4.42l.66.58H14zm-7 4c1.11 0 2-.89 2-2H5c0 1.11.89 2 2 2z"/></svg>
</a>
    </span>
  </li>

  <li class="dropdown">
    <details class="details-overlay details-reset d-flex px-2 flex-items-center">
      <summary class="HeaderNavlink"
         aria-label="Create new…"
         data-ga-click="Header, create new, icon:add">
        <svg class="octicon octicon-plus float-left mr-1 mt-1" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 9H7v5H5V9H0V7h5V2h2v5h5v2z"/></svg>
        <span class="dropdown-caret mt-1"></span>
      </summary>
      <details-menu class="dropdown-menu dropdown-menu-sw">
        
<a role="menuitem" class="dropdown-item" href="/new" data-ga-click="Header, create new repository">
  New repository
</a>

  <a role="menuitem" class="dropdown-item" href="/new/import" data-ga-click="Header, import a repository">
    Import repository
  </a>

<a role="menuitem" class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, create new gist">
  New gist
</a>

  <a role="menuitem" class="dropdown-item" href="/organizations/new" data-ga-click="Header, create new organization">
    New organization
  </a>




      </details-menu>
    </details>
  </li>

  <li class="dropdown">

    <details class="details-overlay details-reset d-flex pl-2 flex-items-center">
      <summary class="HeaderNavlink name mt-1"
        aria-label="View profile and more"
        data-ga-click="Header, show menu, icon:avatar">
        <img alt="@nammbash" class="avatar float-left mr-1" src="https://avatars2.githubusercontent.com/u/38869685?s=40&amp;v=4" height="20" width="20">
        <span class="dropdown-caret"></span>
      </summary>
      <details-menu class="dropdown-menu dropdown-menu-sw">
        <ul>
          <li class="header-nav-current-user css-truncate"><a role="menuitem" class="no-underline user-profile-link px-3 pt-2 pb-2 mb-n2 mt-n1 d-block" href="/nammbash" data-ga-click="Header, go to profile, text:Signed in as">Signed in as <strong class="css-truncate-target">nammbash</strong></a></li>
          <li class="dropdown-divider"></li>


          <li><a role="menuitem" class="dropdown-item" href="/nammbash" data-ga-click="Header, go to profile, text:your profile">Your profile</a></li>
          <li><a role="menuitem" class="dropdown-item" href="/nammbash?tab=repositories" data-ga-click="Header, go to repositories, text:your repositories">Your repositories</a></li>


          <li><a role="menuitem" class="dropdown-item" href="/nammbash?tab=stars" data-ga-click="Header, go to starred repos, text:your stars">Your stars</a></li>
            <li><a role="menuitem" class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, your gists, text:your gists">Your gists</a></li>
          <li class="dropdown-divider"></li>
          <li><a role="menuitem" class="dropdown-item" href="https://help.github.com" data-ga-click="Header, go to help, text:help">Help</a></li>
          <li><a role="menuitem" class="dropdown-item" href="/settings/profile" data-ga-click="Header, go to settings, icon:settings">Settings</a></li>
          <li>
            <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="logout-form" action="/logout" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="52m3ubaEd1eF6IKhSYrvPuKBcmp2SN91EDvw6padH4KPnPV+h5g3tJzB+OG96E6pb9JtuzQPcerkGDj9VNag4g==" />
              <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout" role="menuitem">
                Sign out
              </button>
</form>          </li>
        </ul>
      </details-menu>
    </details>
  </li>
</ul>



        <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="sr-only right-0" action="/logout" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="Iy2VC7Bu5+Zvl0CcLoVeuxNCO1PsMDhAh28RhaxuTURL2NfMgXKnBXa+Otza5/8snhEkgq53lt9zTNmSbiXyJA==" />
          <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout">
            Sign out
          </button>
</form>      </div>
    </div>
  </div>
</header>

      

  </div>

  <div id="start-of-content" class="show-on-focus"></div>

    <div id="js-flash-container">

</div>



  <div role="main" class="application-main " data-commit-hovercards-enabled>
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <div id="js-repo-pjax-container" data-pjax-container >
      


  


  



  <div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav  ">
    <div class="repohead-details-container clearfix container">

      <ul class="pagehead-actions">
  <li>
        <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-remote="true" class="js-social-form js-social-container" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="EeYjpr6GO04x3R/BP6gfvz4y+ZQnwUj+/fCstA5fP2wvh7FsuUNoaz3Y4J7PfeqqgGc3F+WTEdX2wChDXzLy2A==" />      <input type="hidden" name="repository_id" id="repository_id" value="92093330" class="form-control" />

      <details class="details-reset details-overlay select-menu float-left">
        <summary class="btn btn-sm btn-with-count select-menu-button" data-ga-click="Repository, click Watch settings, action:blob#show">
          <span data-menu-button>
              <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
              Watch
          </span>
        </summary>
        <details-menu class="select-menu-modal position-absolute mt-5" style="z-index: 99;">
          <div class="select-menu-header">
            <span class="select-menu-title">Notifications</span>
          </div>
          <div class="select-menu-list">
            <button type="submit" name="do" value="included" class="select-menu-item width-full" aria-checked="true" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Not watching</span>
                <span class="description">Be notified only when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                  Watch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="release_only" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Releases only</span>
                <span class="description">Be notified of new releases, and when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                  Unwatch releases
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="subscribed" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Watching</span>
                <span class="description">Be notified of all conversations.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                  Unwatch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="ignore" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Ignoring</span>
                <span class="description">Never be notified.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-mute v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 2.81v10.38c0 .67-.81 1-1.28.53L3 10H1c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1h2l3.72-3.72C7.19 1.81 8 2.14 8 2.81zm7.53 3.22l-1.06-1.06-1.97 1.97-1.97-1.97-1.06 1.06L11.44 8 9.47 9.97l1.06 1.06 1.97-1.97 1.97 1.97 1.06-1.06L13.56 8l1.97-1.97z"/></svg>
                  Stop ignoring
                </span>
              </div>
            </button>
          </div>
        </details-menu>
      </details>
      <a class="social-count js-social-count"
        href="/NervanaSystems/private-tensorflow/watchers"
        aria-label="65 users are watching this repository">
        65
      </a>
</form>
  </li>

  <li>
      <div class="js-toggler-container js-social-container starring-container ">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="starred js-social-form" action="/NervanaSystems/private-tensorflow/unstar" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="V91sJGdvsKG+xT1vmxTTtHzFhkuS2KfYb9gy/89IXfVrojwH7/YwKSVDhuYVLGtJMDlJ4KyOSttPfGwnlPGy6w==" />
      <input type="hidden" name="context" value="repository"></input>
      <button
        type="submit"
        class="btn btn-sm btn-with-count js-toggler-target"
        aria-label="Unstar this repository" title="Unstar NervanaSystems/private-tensorflow"
        data-ga-click="Repository, click unstar button, action:blob#show; text:Unstar">
        <svg class="octicon octicon-star v-align-text-bottom" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>
        Unstar
      </button>
        <a class="social-count js-social-count" href="/NervanaSystems/private-tensorflow/stargazers"
           aria-label="8 users starred this repository">
          8
        </a>
</form>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="unstarred js-social-form" action="/NervanaSystems/private-tensorflow/star" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="qqWwDpNDt7jQARy9acoFhrkJXYcP5Ycz1PBZ7nfPu6esGj7NaLgUkoQPvYXi0V9t85bUR5bpEMvq8Yz4+41weg==" />
      <input type="hidden" name="context" value="repository"></input>
      <button
        type="submit"
        class="btn btn-sm btn-with-count js-toggler-target"
        aria-label="Star this repository" title="Star NervanaSystems/private-tensorflow"
        data-ga-click="Repository, click star button, action:blob#show; text:Star">
        <svg class="octicon octicon-star v-align-text-bottom" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>
        Star
      </button>
        <a class="social-count js-social-count" href="/NervanaSystems/private-tensorflow/stargazers"
           aria-label="8 users starred this repository">
          8
        </a>
</form>  </div>

  </li>

  <li>
          <details class="details-reset details-overlay details-overlay-dark d-inline-block float-left"
            data-deferred-details-content-url="/NervanaSystems/private-tensorflow/fork?fragment=1">
            <summary class="btn btn-sm btn-with-count"
              title="Fork your own copy of NervanaSystems/private-tensorflow to your account"
              data-ga-click="Repository, show fork modal, action:blob#show; text:Fork">
              <svg class="octicon octicon-repo-forked v-align-text-bottom" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
              Fork
            </summary>
            <details-dialog class="anim-fade-in fast Box Box--overlay d-flex flex-column">
              <div class="Box-header">
                <button class="Box-btn-octicon btn-octicon float-right" type="button" aria-label="Close dialog" data-close-dialog>
                  <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
                </button>
                <h3 class="Box-title">Fork private-tensorflow</h3>
              </div>
              <div class="overflow-auto text-center">
                <include-fragment>
                  <div class="octocat-spinner my-3" aria-label="Loading..."></div>
                  <p class="f5 text-gray">If this dialog fails to load, you can visit <a href="/NervanaSystems/private-tensorflow/fork">the fork page</a> directly.</p>
                </include-fragment>
              </div>
            </details-dialog>
          </details>

    <a href="/NervanaSystems/private-tensorflow/network/members" class="social-count"
       aria-label="6 users forked this repository">
      6
    </a>
  </li>
</ul>

      <h1 class="private ">
  <svg class="octicon octicon-lock" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 13H3v-1h1v1zm8-6v7c0 .55-.45 1-1 1H1c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1h1V4c0-2.2 1.8-4 4-4s4 1.8 4 4v2h1c.55 0 1 .45 1 1zM3.8 6h4.41V4c0-1.22-.98-2.2-2.2-2.2-1.22 0-2.2.98-2.2 2.2v2H3.8zM11 7H2v7h9V7zM4 8H3v1h1V8zm0 2H3v1h1v-1z"/></svg>
  <span class="author" itemprop="author"><a class="url fn" rel="author" data-hovercard-type="organization" data-hovercard-url="/orgs/NervanaSystems/hovercard" href="/NervanaSystems">NervanaSystems</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a data-pjax="#js-repo-pjax-container" href="/NervanaSystems/private-tensorflow">private-tensorflow</a></strong>
    <span class="Label Label--outline v-align-middle">Private</span>

</h1>

    </div>
    
<nav class="reponav js-repo-nav js-sidenav-container-pjax container"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
    aria-label="Repository"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a class="js-selected-navigation-item selected reponav-item" itemprop="url" data-hotkey="g c" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /NervanaSystems/private-tensorflow/tree/perchannel-int8-relu6" href="/NervanaSystems/private-tensorflow/tree/perchannel-int8-relu6">
      <svg class="octicon octicon-code" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>


  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a data-hotkey="g p" itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /NervanaSystems/private-tensorflow/pulls" href="/NervanaSystems/private-tensorflow/pulls">
      <svg class="octicon octicon-git-pull-request" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0 0 10 15a1.993 1.993 0 0 0 1-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v6.56A1.993 1.993 0 0 0 2 15a1.993 1.993 0 0 0 1-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">104</span>
      <meta itemprop="position" content="3">
</a>  </span>


    <a data-hotkey="g b" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /NervanaSystems/private-tensorflow/projects" href="/NervanaSystems/private-tensorflow/projects">
      <svg class="octicon octicon-project" viewBox="0 0 15 16" version="1.1" width="15" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      Projects
      <span class="Counter" >3</span>
</a>

    <a class="js-selected-navigation-item reponav-item" data-hotkey="g w" data-selected-links="repo_wiki /NervanaSystems/private-tensorflow/wiki" href="/NervanaSystems/private-tensorflow/wiki">
      <svg class="octicon octicon-book" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3 5h4v1H3V5zm0 3h4V7H3v1zm0 2h4V9H3v1zm11-5h-4v1h4V5zm0 2h-4v1h4V7zm0 2h-4v1h4V9zm2-6v9c0 .55-.45 1-1 1H9.5l-1 1-1-1H2c-.55 0-1-.45-1-1V3c0-.55.45-1 1-1h5.5l1 1 1-1H15c.55 0 1 .45 1 1zm-8 .5L7.5 3H2v9h6V3.5zm7-.5H9.5l-.5.5V12h6V3z"/></svg>
      Wiki
</a>
    <a class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph pulse alerts security /NervanaSystems/private-tensorflow/pulse" href="/NervanaSystems/private-tensorflow/pulse">
      <svg class="octicon octicon-graph" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
      Insights
</a>

</nav>


  </div>

<div class="container new-discussion-timeline experiment-repo-nav  ">
  <div class="repository-content ">

    
    



  
    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/NervanaSystems/private-tensorflow/blob/02faed6a53512187bd02b1a7ce23646ff4df4d59/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc">Permalink</a>

    <!-- blob contrib key: blob_contributors:v21:945fc17ea2199dc92bd44af2d3cdc9f5 -->

    

    <div class="file-navigation">
      
<div class="select-menu branch-select-menu js-menu-container js-select-menu float-left">
  <button class=" btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    title="perchannel-int8-relu6"
    type="button" aria-label="Switch branches or tags" aria-expanded="false" aria-haspopup="true">
      <i>Branch:</i>
      <span class="js-select-button css-truncate-target">perchannel-int…</span>
  </button>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax>

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <svg class="octicon octicon-x js-menu-close" role="img" aria-label="Close" viewBox="0 0 12 16" version="1.1" width="12" height="16"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
        <span class="select-menu-title">Switch branches/tags</span>
      </div>

      <tab-container>
      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Find or create a branch…" id="context-commitish-filter-field" class="form-control js-filterable-field js-navigation-enable" placeholder="Find or create a branch…">
        </div>
        <div class="select-menu-tabs" role="tablist">
          <ul>
            <li class="select-menu-tab">
              <button type="button" class="select-menu-tab-nav" data-filter-placeholder="Find or create a branch…" role="tab" aria-selected="true">Branches</button>
            </li>
            <li class="select-menu-tab">
              <button type="button" class="select-menu-tab-nav" data-filter-placeholder="Find a tag…" role="tab">Tags</button>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list" role="tabpanel">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/3dconv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="3dconv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                3dconv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/AVX512-embedding_ops_test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="AVX512-embedding_ops_test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                AVX512-embedding_ops_test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/AdamOptimize/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="AdamOptimize"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                AdamOptimize
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/Bootstrapping-MKL-test-infrastructure/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="Bootstrapping-MKL-test-infrastructure"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                Bootstrapping-MKL-test-infrastructure
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/Densenet_dropout_removal/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="Densenet_dropout_removal"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                Densenet_dropout_removal
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/Elementwiseopt/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="Elementwiseopt"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                Elementwiseopt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/UpdateBlocktime/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="UpdateBlocktime"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                UpdateBlocktime
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/VNNI_DUNGEON_WW49_FIX_Weight_Cache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="VNNI_DUNGEON_WW49_FIX_Weight_Cache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                VNNI_DUNGEON_WW49_FIX_Weight_Cache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/VNNI_DUNGEON_WW49_FIX/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="VNNI_DUNGEON_WW49_FIX"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                VNNI_DUNGEON_WW49_FIX
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/VNNI_Dungeon_FP32_Internal_WIP_Weight_Cache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="VNNI_Dungeon_FP32_Internal_WIP_Weight_Cache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                VNNI_Dungeon_FP32_Internal_WIP_Weight_Cache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/VNNI_Dungeon_FP32_Internal_WIP/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="VNNI_Dungeon_FP32_Internal_WIP"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                VNNI_Dungeon_FP32_Internal_WIP
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/VNNI_Dungeon_FP32_Internal_with_INT8FunctionalityFix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="VNNI_Dungeon_FP32_Internal_with_INT8FunctionalityFix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                VNNI_Dungeon_FP32_Internal_with_INT8FunctionalityFix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/VNNI_fp32_int8_merged/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="VNNI_fp32_int8_merged"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                VNNI_fp32_int8_merged
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/accuracy_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="accuracy_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                accuracy_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/add-cc-tests/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="add-cc-tests"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                add-cc-tests
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/StringPiece_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/StringPiece_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/StringPiece_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/alt_pool_v2_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/alt_pool_v2_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/alt_pool_v2_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/conv-relufusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/conv-relufusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/conv-relufusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/conv_bug_fixes/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/conv_bug_fixes"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/conv_bug_fixes
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/disable-ml/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/disable-ml"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/disable-ml
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/disable_mkl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/disable_mkl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/disable_mkl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/disable_transpose_3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/disable_transpose_3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/disable_transpose_3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/grappler_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/grappler_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/grappler_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/klockwork-fixes/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/klockwork-fixes"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/klockwork-fixes
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/license_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/license_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/license_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/mkl_layout_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/mkl_layout_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/mkl_layout_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/mkl_ml_update/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/mkl_ml_update"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/mkl_ml_update
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/model_prfoiler_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/model_prfoiler_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/model_prfoiler_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/pooling_v2fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/pooling_v2fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/pooling_v2fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/remove_mkl_only_matmul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/remove_mkl_only_matmul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/remove_mkl_only_matmul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/remove_ml_misc_ops/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/remove_ml_misc_ops"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/remove_ml_misc_ops
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/tmp_reuse_disable/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/tmp_reuse_disable"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/tmp_reuse_disable
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/tmp/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/tmp"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/tmp
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/agramesh/tmpfusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="agramesh/tmpfusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                agramesh/tmpfusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/amazon_patch/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="amazon_patch"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                amazon_patch
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/amin/perchannel-convs/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="amin/perchannel-convs"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                amin/perchannel-convs
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/PadWithConv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/PadWithConv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/PadWithConv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/bug83/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/bug83"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/bug83
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/bug236/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/bug236"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/bug236
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/bugfix_directSession/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/bugfix_directSession"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/bugfix_directSession
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/int8-depthwise/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/int8-depthwise"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/int8-depthwise
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/lrn-remove-mkl-ml/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/lrn-remove-mkl-ml"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/lrn-remove-mkl-ml
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/mkldnn_layout/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/mkldnn_layout"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/mkldnn_layout
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/padwithconv_latest_opt/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/padwithconv_latest_opt"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/padwithconv_latest_opt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/perchannel-int8-relu6-depthwise-merged/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/perchannel-int8-relu6-depthwise-merged"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/perchannel-int8-relu6-depthwise-merged
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/perchannel-int8-relu6-depthwise/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/perchannel-int8-relu6-depthwise"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/perchannel-int8-relu6-depthwise
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/quantize/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/quantize"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/quantize
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/relu-remove-mkl-ml/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/relu-remove-mkl-ml"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/relu-remove-mkl-ml
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/slice/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/slice"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/slice
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/softmax-remove-mkl-ml/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/softmax-remove-mkl-ml"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/softmax-remove-mkl-ml
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ashraf/softmax_fp32/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ashraf/softmax_fp32"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ashraf/softmax_fp32
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/avx-sgemm-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="avx-sgemm-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                avx-sgemm-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/backup-mkldnn-quantization-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="backup-mkldnn-quantization-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                backup-mkldnn-quantization-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bfc-alloc/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bfc-alloc"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bfc-alloc
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/fix-build-failure/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/fix-build-failure"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/fix-build-failure
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/matmul-bug-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/matmul-bug-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/matmul-bug-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/quantized-conv-pad-fusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/quantized-conv-pad-fusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/quantized-conv-pad-fusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/tfdo-178-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/tfdo-178-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/tfdo-178-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/upgrade-mkldnn-0.16/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/upgrade-mkldnn-0.16"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/upgrade-mkldnn-0.16
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/weight-caching-latest/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/weight-caching-latest"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/weight-caching-latest
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bhavanis/weight-caching/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bhavanis/weight-caching"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bhavanis/weight-caching
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bugfix_directSession/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bugfix_directSession"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bugfix_directSession
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/bugfix/wenxizhu/transpose-removal/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="bugfix/wenxizhu/transpose-removal"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bugfix/wenxizhu/transpose-removal
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/build_issue_tf1.9/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="build_issue_tf1.9"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                build_issue_tf1.9
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/changeinteop/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="changeinteop"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                changeinteop
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/code_cleanup/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="code_cleanup"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                code_cleanup
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/concat-primitive-creation-bug-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="concat-primitive-creation-bug-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                concat-primitive-creation-bug-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/concat_build_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="concat_build_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                concat_build_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/concat_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="concat_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                concat_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/control-dependency-in-fusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="control-dependency-in-fusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                control-dependency-in-fusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/cosim-sh/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="cosim-sh"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cosim-sh
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/cui_patch_pr/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="cui_patch_pr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cui_patch_pr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/cuixiaom_matmul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="cuixiaom_matmul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cuixiaom_matmul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/cuixiaom_mst_dismkl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="cuixiaom_mst_dismkl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cuixiaom_mst_dismkl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/cuixiaom_pr/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="cuixiaom_pr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cuixiaom_pr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/cuixiaom_update_dismkl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="cuixiaom_update_dismkl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cuixiaom_update_dismkl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/daniel1-martin/example/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="daniel1-martin/example"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                daniel1-martin/example
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/daniel1-martin/latest_opts_int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="daniel1-martin/latest_opts_int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                daniel1-martin/latest_opts_int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/daniel1-martin/pooling-into-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="daniel1-martin/pooling-into-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                daniel1-martin/pooling-into-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/daoxin/TransposeReorderCache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="daoxin/TransposeReorderCache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                daoxin/TransposeReorderCache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/datalayer/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="datalayer"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                datalayer
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/delock/optimize_filter_with_random_uniform/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="delock/optimize_filter_with_random_uniform"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                delock/optimize_filter_with_random_uniform
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/depthwise_conv2D/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="depthwise_conv2D"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                depthwise_conv2D
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/disable_mkl_default/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="disable_mkl_default"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                disable_mkl_default
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/dnn_v014_migration/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="dnn_v014_migration"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dnn_v014_migration
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/do_for_mkldnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="do_for_mkldnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                do_for_mkldnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/dummytest/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="dummytest"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dummytest
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/experiment/opt-for-3dunet/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="experiment/opt-for-3dunet"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                experiment/opt-for-3dunet
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/experiment/wenxizhu/bn_relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="experiment/wenxizhu/bn_relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                experiment/wenxizhu/bn_relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/daoxin/cropAndResizeParallize/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/daoxin/cropAndResizeParallize"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/daoxin/cropAndResizeParallize
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/daoxin/slice/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/daoxin/slice"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/daoxin/slice
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/daoxin/sliceReorderCache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/daoxin/sliceReorderCache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/daoxin/sliceReorderCache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/daoxin/sliceReorderCacheNew/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/daoxin/sliceReorderCacheNew"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/daoxin/sliceReorderCacheNew
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/tenglu/fix_intra_pool/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/tenglu/fix_intra_pool"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/tenglu/fix_intra_pool
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/tenglu/fuse_matmul_with_bias/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/tenglu/fuse_matmul_with_bias"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/tenglu/fuse_matmul_with_bias
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/wenxi/maxpool3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/wenxi/maxpool3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/wenxi/maxpool3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/wenxi/redundant-transpose-removal-for-conv2d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/wenxi/redundant-transpose-removal-for-conv2d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/wenxi/redundant-transpose-removal-for-conv2d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/wenxi/redundant-transpose-removal-for-conv3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/wenxi/redundant-transpose-removal-for-conv3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/wenxi/redundant-transpose-removal-for-conv3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/wenxi/redundant-transpose-removal-for-deconv3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/wenxi/redundant-transpose-removal-for-deconv3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/wenxi/redundant-transpose-removal-for-deconv3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/wenxi/redundant-transpose-removal-for-maxpool3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/wenxi/redundant-transpose-removal-for-maxpool3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/wenxi/redundant-transpose-removal-for-maxpool3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/wenxizhu/conv_biasadd_relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/wenxizhu/conv_biasadd_relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/wenxizhu/conv_biasadd_relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/conv_3d_batchnorm/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/conv_3d_batchnorm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/conv_3d_batchnorm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/conv_3d_bwd/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/conv_3d_bwd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/conv_3d_bwd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/conv_3d_fwd/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/conv_3d_fwd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/conv_3d_fwd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/conv_3d_pooling/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/conv_3d_pooling"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/conv_3d_pooling
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/conv_3d_relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/conv_3d_relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/conv_3d_relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/neg_op/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/neg_op"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/neg_op
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/feature/zhoulong/relu_3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="feature/zhoulong/relu_3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                feature/zhoulong/relu_3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/fix-non-mkl-builds/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="fix-non-mkl-builds"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fix-non-mkl-builds
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/fix_mkl_input_conversion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="fix_mkl_input_conversion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fix_mkl_input_conversion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/fix_typo_env/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="fix_typo_env"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fix_typo_env
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/fixed_nhwc_resnet/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="fixed_nhwc_resnet"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fixed_nhwc_resnet
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/fixing-build-issue/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="fixing-build-issue"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fixing-build-issue
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/fixurl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="fixurl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fixurl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gomptest/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gomptest"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gomptest
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili_addn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili_addn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili_addn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili_depthwise_conv_bwd/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili_depthwise_conv_bwd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili_depthwise_conv_bwd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili_depthwise_conv_bwd_1/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili_depthwise_conv_bwd_1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili_depthwise_conv_bwd_1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili/leakyrelu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili/leakyrelu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili/leakyrelu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili/pad_fusedconv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili/pad_fusedconv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili/pad_fusedconv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili_relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili_relu6"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili_relu6
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili/update_mkldnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili/update_mkldnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili/update_mkldnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/guizili/update_mklml/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="guizili/update_mklml"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                guizili/update_mklml
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/conv3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/conv3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/conv3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/eps/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/eps"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/eps
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/extrap/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/extrap"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/extrap
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/fix_typo_env/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/fix_typo_env"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/fix_typo_env
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/graph_transform_mul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/graph_transform_mul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/graph_transform_mul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/identity/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/identity"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/identity
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/lrucache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/lrucache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/lrucache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/mkldnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/mkldnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/mkldnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/mklml_cleanup_batchnorm/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/mklml_cleanup_batchnorm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/mklml_cleanup_batchnorm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/mklml_cleanup_conv_bwd/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/mklml_cleanup_conv_bwd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/mklml_cleanup_conv_bwd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/prim_reuse_limit_caching/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/prim_reuse_limit_caching"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/prim_reuse_limit_caching
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/primitive_reuse/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/primitive_reuse"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/primitive_reuse
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/qtool_init/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/qtool_init"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/qtool_init
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/qtools_channelwise/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/qtools_channelwise"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/qtools_channelwise
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/qtools_fold_subdivmul_bn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/qtools_fold_subdivmul_bn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/qtools_fold_subdivmul_bn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/qtools_pad/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/qtools_pad"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/qtools_pad
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz/qtools_quantize/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz/qtools_quantize"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz/qtools_quantize
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz2_layout/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz2_layout"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz2_layout
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/gz_2/mkldnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="gz_2/mkldnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                gz_2/mkldnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/hfei3/concat_remove_ml_only/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="hfei3/concat_remove_ml_only"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                hfei3/concat_remove_ml_only
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/hfei3/int8-graph-transform/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="hfei3/int8-graph-transform"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                hfei3/int8-graph-transform
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/in8-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="in8-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                in8-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-bug-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-bug-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-bug-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-conv-and-common-code/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-conv-and-common-code"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-conv-and-common-code
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-conv-part1/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-conv-part1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-conv-part1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-conv-part2/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-conv-part2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-conv-part2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-conv-part3/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-conv-part3"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-conv-part3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-conv-part4/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-conv-part4"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-conv-part4
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-frcnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-frcnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-frcnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-inference-example/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-inference-example"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-inference-example
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-master-after-merge/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-master-after-merge"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-master-after-merge
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-master-and-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-master-and-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-master-and-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/int8-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="int8-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                int8-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/intel-quantization-tools/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="intel-quantization-tools"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                intel-quantization-tools
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/intel/mkl_singularity/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="intel/mkl_singularity"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                intel/mkl_singularity
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jbobba/batchmatmul_mkl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jbobba/batchmatmul_mkl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jbobba/batchmatmul_mkl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jbobba/cwise_mulopt/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jbobba/cwise_mulopt"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jbobba/cwise_mulopt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jbobba/mkl-allocator-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jbobba/mkl-allocator-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jbobba/mkl-allocator-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jbobba/mkl_batchmatmul_PR/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jbobba/mkl_batchmatmul_PR"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jbobba/mkl_batchmatmul_PR
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jbobba/numa/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jbobba/numa"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jbobba/numa
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jing/addn_all/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jing/addn_all"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jing/addn_all
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jing/addn_fixs/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jing/addn_fixs"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jing/addn_fixs
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jing/bug224/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jing/bug224"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jing/bug224
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jing/cache-weight/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jing/cache-weight"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jing/cache-weight
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jing/conv+relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jing/conv+relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jing/conv+relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jing/dilated_conv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jing/dilated_conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jing/dilated_conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/jinghua2/dnn-addn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="jinghua2/dnn-addn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jinghua2/dnn-addn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/joyce/test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="joyce/test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                joyce/test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/karen/testbuild/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="karen/testbuild"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                karen/testbuild
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/keras-packages/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="keras-packages"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                keras-packages
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/klockwork-TFDO-504/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="klockwork-TFDO-504"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                klockwork-TFDO-504
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/klockwork-tfdo-501/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="klockwork-tfdo-501"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                klockwork-tfdo-501
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/klockwork-tfdo-506/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="klockwork-tfdo-506"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                klockwork-tfdo-506
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/klockwork-tfdo-1152/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="klockwork-tfdo-1152"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                klockwork-tfdo-1152
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/klockworks-TFDO-509/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="klockworks-TFDO-509"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                klockworks-TFDO-509
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/lakshay/depthwise/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="lakshay/depthwise"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lakshay/depthwise
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/last/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="last"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                last
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/latest_optimizations_with_latest_mkldnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="latest_optimizations_with_latest_mkldnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                latest_optimizations_with_latest_mkldnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/latest_optimizations_with_primReorderReuse/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="latest_optimizations_with_primReorderReuse"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                latest_optimizations_with_primReorderReuse
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/latest_optimizations_with_reorder/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="latest_optimizations_with_reorder"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                latest_optimizations_with_reorder
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/latest_optimizations/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="latest_optimizations"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                latest_optimizations
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/lto/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="lto"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lto
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/ltokas/alloc_test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="ltokas/alloc_test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ltokas/alloc_test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mabuzain/int8-pad-fusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mabuzain/int8-pad-fusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mabuzain/int8-pad-fusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mabuzain/int8-per-channel/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mabuzain/int8-per-channel"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mabuzain/int8-per-channel
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mabuzain/klocwork-fixes/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mabuzain/klocwork-fixes"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mabuzain/klocwork-fixes
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mabuzain/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mabuzain/perchannel-int8-relu6"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mabuzain/perchannel-int8-relu6
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mandy/test-openmp-settings/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mandy/test-openmp-settings"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mandy/test-openmp-settings
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mandy/upgrade-mkldnn-version/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mandy/upgrade-mkldnn-version"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mandy/upgrade-mkldnn-version
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mandy/upgrade_mkldnn_version/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mandy/upgrade_mkldnn_version"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mandy/upgrade_mkldnn_version
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/markdryan-LRNGrad-pow/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="markdryan-LRNGrad-pow"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                markdryan-LRNGrad-pow
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/markdryan-avx512-nntest/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="markdryan-avx512-nntest"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                markdryan-avx512-nntest
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/bias-add/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/bias-add"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/bias-add
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/debug-quantization/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/debug-quantization"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/debug-quantization
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/eigen-avx512-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/eigen-avx512-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/eigen-avx512-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/eigen-bias-add/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/eigen-bias-add"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/eigen-bias-add
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/expose-mkl-op/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/expose-mkl-op"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/expose-mkl-op
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/fake-bias-add-grad/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/fake-bias-add-grad"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/fake-bias-add-grad
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/fp32-fusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/fp32-fusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/fp32-fusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/fuse-requant/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/fuse-requant"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/fuse-requant
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/fused-filter-bias-grad/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/fused-filter-bias-grad"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/fused-filter-bias-grad
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/graph-transform/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/graph-transform"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/graph-transform
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/inplace-relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/inplace-relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/inplace-relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/mkldnn-qconv-fp/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/mkldnn-qconv-fp"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/mkldnn-qconv-fp
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/mkldnn-quantized-conv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/mkldnn-quantized-conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/mkldnn-quantized-conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/mkldnn-reshape/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/mkldnn-reshape"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/mkldnn-reshape
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/quant-conv-bias-relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/quant-conv-bias-relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/quant-conv-bias-relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/quant-fused-conv-all/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/quant-fused-conv-all"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/quant-fused-conv-all
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/quant-fused-conv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/quant-fused-conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/quant-fused-conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mdfaijul/reshape-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mdfaijul/reshape-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mdfaijul/reshape-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkl-dnn-open-source-only/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkl-dnn-open-source-only"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkl-dnn-open-source-only
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkl-graph-viz/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkl-graph-viz"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkl-graph-viz
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkl_registration_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkl_registration_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkl_registration_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkl_slice_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkl_slice_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkl_slice_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkldnn-int8-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkldnn-int8-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkldnn-int8-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkldnn-quantization-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkldnn-quantization-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkldnn-quantization-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mkldnn-transposend/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mkldnn-transposend"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mkldnn-transposend
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mobssd-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mobssd-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mobssd-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/mtcnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="mtcnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mtcnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/new_fix_mkl_input_conversion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="new_fix_mkl_input_conversion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                new_fix_mkl_input_conversion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/nhasabni/graphpassforall/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="nhasabni/graphpassforall"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nhasabni/graphpassforall
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/nhasabni/mkldnn-conv-eltwise-fusion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="nhasabni/mkldnn-conv-eltwise-fusion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nhasabni/mkldnn-conv-eltwise-fusion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/nhasabni/mkldnn-slice/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="nhasabni/mkldnn-slice"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nhasabni/mkldnn-slice
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/nhasabni/mkldnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="nhasabni/mkldnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nhasabni/mkldnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/nhasabni_srini/mklml-pooling-cleanup/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="nhasabni_srini/mklml-pooling-cleanup"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nhasabni_srini/mklml-pooling-cleanup
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/nhasabni/transpose/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="nhasabni/transpose"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nhasabni/transpose
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/niroop/fold_subdivmul_batchnorms/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="niroop/fold_subdivmul_batchnorms"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                niroop/fold_subdivmul_batchnorms
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/niroop/int8-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="niroop/int8-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                niroop/int8-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/niroop/s8s8s32-int8-master/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="niroop/s8s8s32-int8-master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                niroop/s8s8s32-int8-master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/numa-ht/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="numa-ht"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                numa-ht
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/old_optimizations/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="old_optimizations"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                old_optimizations
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/openMP-test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="openMP-test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                openMP-test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/opencl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="opencl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                opencl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/origin/feature/zhoulong/relu_3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="origin/feature/zhoulong/relu_3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                origin/feature/zhoulong/relu_3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/patch_1.8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="patch_1.8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                patch_1.8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/perchannel-int8-relu6-depthwise/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="perchannel-int8-relu6-depthwise"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perchannel-int8-relu6-depthwise
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/NervanaSystems/private-tensorflow/blob/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="perchannel-int8-relu6"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perchannel-int8-relu6
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/perchannel-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="perchannel-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perchannel-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pooling-and-addn-fixes/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pooling-and-addn-fixes"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pooling-and-addn-fixes
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pooling-bug-fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pooling-bug-fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pooling-bug-fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pr-adding-license-file/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pr-adding-license-file"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pr-adding-license-file
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pr-fix-util_cuda_kernel_helper_test_gpu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pr-fix-util_cuda_kernel_helper_test_gpu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pr-fix-util_cuda_kernel_helper_test_gpu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pr-making-mkl-dnn-default-build-choice-2/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pr-making-mkl-dnn-default-build-choice-2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pr-making-mkl-dnn-default-build-choice-2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pr-making-mkl-dnn-default-build-choice/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pr-making-mkl-dnn-default-build-choice"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pr-making-mkl-dnn-default-build-choice
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/pr_fix_api_compatibility_test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="pr_fix_api_compatibility_test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                pr_fix_api_compatibility_test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/prim_reuse_disable/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="prim_reuse_disable"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                prim_reuse_disable
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_batch_norm/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_batch_norm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_batch_norm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_conv_bwd/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_conv_bwd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_conv_bwd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_conv_fwd_update/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_conv_fwd_update"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_conv_fwd_update
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_conv_fwd/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_conv_fwd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_conv_fwd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_pooling/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_pooling"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_pooling
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_relu/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_relu"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_relu
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_reorder/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_reorder"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_reorder
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_temp_disable/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_temp_disable"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_temp_disable
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_working_rebase/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_working_rebase"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_working_rebase
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/primreuse_working/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="primreuse_working"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                primreuse_working
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/py36whl/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="py36whl"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                py36whl
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/quantized_concat/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="quantized_concat"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                quantized_concat
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/quantized_pooling/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="quantized_pooling"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                quantized_pooling
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/r1.8_disMKLDNN/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="r1.8_disMKLDNN"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                r1.8_disMKLDNN
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/r1.8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="r1.8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                r1.8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/r1.11/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="r1.11"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                r1.11
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/refine-mkl-gpu-test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="refine-mkl-gpu-test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                refine-mkl-gpu-test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/relu_bn_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="relu_bn_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                relu_bn_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/relu_build_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="relu_build_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                relu_build_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/reorder-reuse-convback/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="reorder-reuse-convback"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                reorder-reuse-convback
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/revert-383-daoxin/TransposeReorderCache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="revert-383-daoxin/TransposeReorderCache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                revert-383-daoxin/TransposeReorderCache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/self_debug_sh/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="self_debug_sh"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                self_debug_sh
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/set_omp/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="set_omp"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                set_omp
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu/tensor_dump/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu/tensor_dump"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu/tensor_dump
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/BUG-251/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/BUG-251"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/BUG-251
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/bug-input-conversion/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/bug-input-conversion"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/bug-input-conversion
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/cache-weight/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/cache-weight"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/cache-weight
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/check-positive-mul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/check-positive-mul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/check-positive-mul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/const/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/const"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/const
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/cosim-sh/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/cosim-sh"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/cosim-sh
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/dilate/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/dilate"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/dilate
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/dilated_conv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/dilated_conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/dilated_conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/dump-graph/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/dump-graph"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/dump-graph
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/eigen_vml/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/eigen_vml"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/eigen_vml
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/fuse-pad-conv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/fuse-pad-conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/fuse-pad-conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/fuse_requantize/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/fuse_requantize"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/fuse_requantize
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/int8-internal/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/int8-internal"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/int8-internal
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/int8-tensor-dump/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/int8-tensor-dump"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/int8-tensor-dump
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/int8-test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/int8-test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/int8-test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/ipp/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/ipp"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/ipp
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/jpeg/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/jpeg"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/jpeg
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/matmul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/matmul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/matmul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/mkl_dnn_icc/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/mkl_dnn_icc"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/mkl_dnn_icc
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/mkldnn-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/mkldnn-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/mkldnn-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/numa-device/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/numa-device"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/numa-device
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/numa-ht/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/numa-ht"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/numa-ht
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/numa-memory/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/numa-memory"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/numa-memory
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/numa-pr/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/numa-pr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/numa-pr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/omp-mix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/omp-mix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/omp-mix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/ompthreadpool/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/ompthreadpool"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/ompthreadpool
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/primitive-reuse-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/primitive-reuse-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/primitive-reuse-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/quantize_transform/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/quantize_transform"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/quantize_transform
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/quantized_concat/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/quantized_concat"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/quantized_concat
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/quantized_pooling/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/quantized_pooling"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/quantized_pooling
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/resize/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/resize"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/resize
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/ssd-vgg16/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/ssd-vgg16"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/ssd-vgg16
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/tensor-dump-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/tensor-dump-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/tensor-dump-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/tensor_dump/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/tensor_dump"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/tensor_dump
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/unit_test/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/unit_test"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/unit_test
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/unittest/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/unittest"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/unittest
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sfu2/vtune/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sfu2/vtune"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sfu2/vtune
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sgemm_dnn_merge/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sgemm_dnn_merge"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sgemm_dnn_merge
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sgemm_dnn15/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sgemm_dnn15"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sgemm_dnn15
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/shaohua/fix_convet_ut/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="shaohua/fix_convet_ut"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                shaohua/fix_convet_ut
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/shaohua/fix_gcc6.3_build_link_issue/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="shaohua/fix_gcc6.3_build_link_issue"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                shaohua/fix_gcc6.3_build_link_issue
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/shwetaoj/mkl_dockerfile_python3/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="shwetaoj/mkl_dockerfile_python3"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                shwetaoj/mkl_dockerfile_python3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sparseMatmul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sparseMatmul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sparseMatmul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sriniva2/densenet_tr/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sriniva2/densenet_tr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sriniva2/densenet_tr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sriniva2/gcp_allocator_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sriniva2/gcp_allocator_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sriniva2/gcp_allocator_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sriniva2/mkl-dnn-sgemm/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sriniva2/mkl-dnn-sgemm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sriniva2/mkl-dnn-sgemm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sriniva2/small_alloc_simple/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sriniva2/small_alloc_simple"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sriniva2/small_alloc_simple
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sriniva2/softmax_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sriniva2/softmax_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sriniva2/softmax_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/sriniva2/stringpiece_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="sriniva2/stringpiece_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sriniva2/stringpiece_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/support_Mac_Build/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="support_Mac_Build"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                support_Mac_Build
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/tenglu/parallelize_UnsortedSegmentSum/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="tenglu/parallelize_UnsortedSegmentSum"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                tenglu/parallelize_UnsortedSegmentSum
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/tensorflow/tools/sigularity/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="tensorflow/tools/sigularity"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                tensorflow/tools/sigularity
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/test-JIRA/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="test-JIRA"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                test-JIRA
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/test-branch-pr/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="test-branch-pr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                test-branch-pr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/testing-external-pr-19008/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="testing-external-pr-19008"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                testing-external-pr-19008
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/testing-pr-upgrade-mkl-dnn/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="testing-pr-upgrade-mkl-dnn"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                testing-pr-upgrade-mkl-dnn
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/tf-1.11/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="tf-1.11"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                tf-1.11
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/tlokas/allo/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="tlokas/allo"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                tlokas/allo
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/tlokas/alloc/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="tlokas/alloc"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                tlokas/alloc
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/update-1.10-README.md/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="update-1.10-README.md"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                update-1.10-README.md
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/update_mkldnn_v13/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="update_mkldnn_v13"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                update_mkldnn_v13
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/updating-to-MKL-DNN-v0.15/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="updating-to-MKL-DNN-v0.15"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                updating-to-MKL-DNN-v0.15
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/upgrade_to_mkldnn_0.15/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="upgrade_to_mkldnn_0.15"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                upgrade_to_mkldnn_0.15
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/v1.3.0-rc2/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="v1.3.0-rc2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                v1.3.0-rc2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/vrane/bug_fixes/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="vrane/bug_fixes"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                vrane/bug_fixes
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/vrane/mkldnn-elementwise/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="vrane/mkldnn-elementwise"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                vrane/mkldnn-elementwise
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/vrane/mkldnn-int8/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="vrane/mkldnn-int8"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                vrane/mkldnn-int8
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/vrane/mkldnn-matmul/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="vrane/mkldnn-matmul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                vrane/mkldnn-matmul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/wenxizhu/maxpooling3d/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="wenxizhu/maxpooling3d"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                wenxizhu/maxpooling3d
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/wenxizhu/transpose-elimination/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="wenxizhu/transpose-elimination"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                wenxizhu/transpose-elimination
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/yli135/concat_fix/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="yli135/concat_fix"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                yli135/concat_fix
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/yli135/depthwise_conv/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="yli135/depthwise_conv"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                yli135/depthwise_conv
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/yli135/fix_bn_memory_leak/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="yli135/fix_bn_memory_leak"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                yli135/fix_bn_memory_leak
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/zhoulong/cache_weight/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="zhoulong/cache_weight"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                zhoulong/cache_weight
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/zhoulong/mkl_switch_op/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="zhoulong/mkl_switch_op"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                zhoulong/mkl_switch_op
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/zhoulong/primreuse_batch_norm_reorderCache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="zhoulong/primreuse_batch_norm_reorderCache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                zhoulong/primreuse_batch_norm_reorderCache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/zhoulong/primreuse_pooling_reorderCache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="zhoulong/primreuse_pooling_reorderCache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                zhoulong/primreuse_pooling_reorderCache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/zhoulong/primreuse_relu_reorder_cache/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="zhoulong/primreuse_relu_reorder_cache"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                zhoulong/primreuse_relu_reorder_cache
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/NervanaSystems/private-tensorflow/blob/zhoulong/switch_merge_dev/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc"
               data-name="zhoulong/switch_merge_dev"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                zhoulong/switch_merge_dev
              </span>
            </a>
        </div>

          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="select-menu-new-item-form js-new-item-form" action="/NervanaSystems/private-tensorflow/branches" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="TxUaqMPcWHXBKaZYqntBNCcpgMjiFw9sLONwlh8nqgS9TCS6MyDWaG8cFhj3x8Y7m0+JchkJvlkI5vpdsvB86Q==" />
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="perchannel-int8-relu6">
            <input type="hidden" name="path_binary" id="path_binary" value="dGVuc29yZmxvdy9jb3JlL2tlcm5lbHMvbWtsX3JlcXVhbnRpemF0aW9uX3JhbmdlX3Blcl9jaGFubmVsX29wLmNj">

            <button type="submit" class="width-full select-menu-item js-navigation-open js-navigation-item">
              <svg class="octicon octicon-git-branch select-menu-item-icon" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M10 5c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v.3c-.02.52-.23.98-.63 1.38-.4.4-.86.61-1.38.63-.83.02-1.48.16-2 .45V4.72a1.993 1.993 0 0 0-1-3.72C.88 1 0 1.89 0 3a2 2 0 0 0 1 1.72v6.56c-.59.35-1 .99-1 1.72 0 1.11.89 2 2 2 1.11 0 2-.89 2-2 0-.53-.2-1-.53-1.36.09-.06.48-.41.59-.47.25-.11.56-.17.94-.17 1.05-.05 1.95-.45 2.75-1.25S8.95 7.77 9 6.73h-.02C9.59 6.37 10 5.73 10 5zM2 1.8c.66 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2C1.35 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2zm0 12.41c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm6-8c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Create branch: <span class="js-new-item-name"></span></span>
                <span class="description">from ‘perchannel-int8-relu6’</span>
              </div>
            </button>
</form>
      </div>

      <div class="select-menu-list" role="tabpanel" hidden>
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>
      </tab-container>
    </div>
  </div>
</div>

      <div class="BtnGroup float-right">
        <a href="/NervanaSystems/private-tensorflow/find/perchannel-int8-relu6"
              class="js-pjax-capture-input btn btn-sm BtnGroup-item"
              data-pjax
              data-hotkey="t">
          Find file
        </a>
        <clipboard-copy for="blob-path" class="btn btn-sm BtnGroup-item">
          Copy path
        </clipboard-copy>
      </div>
      <div id="blob-path" class="breadcrumb">
        <span class="repo-root js-repo-root"><span class="js-path-segment"><a data-pjax="true" href="/NervanaSystems/private-tensorflow/tree/perchannel-int8-relu6"><span>private-tensorflow</span></a></span></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/NervanaSystems/private-tensorflow/tree/perchannel-int8-relu6/tensorflow"><span>tensorflow</span></a></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/NervanaSystems/private-tensorflow/tree/perchannel-int8-relu6/tensorflow/core"><span>core</span></a></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/NervanaSystems/private-tensorflow/tree/perchannel-int8-relu6/tensorflow/core/kernels"><span>kernels</span></a></span><span class="separator">/</span><strong class="final-path">mkl_requantization_range_per_channel_op.cc</strong>
      </div>
    </div>


    <include-fragment src="/NervanaSystems/private-tensorflow/contributors/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc" class="commit-tease commit-loader">
      <div>
        Fetching contributors&hellip;
      </div>

      <div class="commit-tease-contributors">
          <img alt="" class="loader-loading float-left" src="https://github.githubassets.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" height="16" />
        <span class="loader-error">Cannot retrieve contributors at this time</span>
      </div>
</include-fragment>


    <div class="file ">
      
<div class="file-header">

  <div class="file-actions">


    <div class="BtnGroup">
      <a id="raw-url" class="btn btn-sm BtnGroup-item" href="/NervanaSystems/private-tensorflow/raw/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc">Raw</a>
        <a class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b" href="/NervanaSystems/private-tensorflow/blame/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc">Blame</a>
      <a rel="nofollow" class="btn btn-sm BtnGroup-item" href="/NervanaSystems/private-tensorflow/commits/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc">History</a>
    </div>


          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form js-update-url-with-hash" action="/NervanaSystems/private-tensorflow/edit/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="ae+4kzgVUQP1nH6Ky4Hht7250dFglOGLViN/cG4jJpRNR0Jaesvlx0oZIPVRq7/ZcsEVditNc/umSzs97/Pnsg==" />
            <button class="btn-octicon tooltipped tooltipped-nw" type="submit"
              aria-label="Edit this file" data-hotkey="e" data-disable-with>
              <svg class="octicon octicon-pencil" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
            </button>
</form>
        <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form" action="/NervanaSystems/private-tensorflow/delete/perchannel-int8-relu6/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="Y3wB7htY0vrXdj9vcFMkhi5UoDXQ8B6l8KgC1pGb3cXIBtQa3nM05T0rEKEBqLjFtjQPC3X1G9gKlUyN4bex2w==" />
          <button class="btn-octicon btn-octicon-danger tooltipped tooltipped-nw" type="submit"
            aria-label="Delete this file" data-disable-with>
            <svg class="octicon octicon-trashcan" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
          </button>
</form>  </div>

  <div class="file-info">
      119 lines (97 sloc)
      <span class="file-info-divider"></span>
    4.46 KB
  </div>
</div>

      

  <div itemprop="text" class="blob-wrapper data type-c ">
      
<table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">/*</span> Copyright 2015 The TensorFlow Authors. All Rights Reserved.</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class="pl-c"></span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class="pl-c">Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line"><span class="pl-c">you may not use this file except in compliance with the License.</span></td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class="pl-c">You may obtain a copy of the License at</span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class="pl-c"></span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class="pl-c">    http://www.apache.org/licenses/LICENSE-2.0</span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class="pl-c"></span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line"><span class="pl-c">Unless required by applicable law or agreed to in writing, software</span></td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line"><span class="pl-c">distributed under the License is distributed on an &quot;AS IS&quot; BASIS,</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class="pl-c">WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.</span></td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line"><span class="pl-c">See the License for the specific language governing permissions and</span></td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line"><span class="pl-c">limitations under the License.</span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line"><span class="pl-c">==============================================================================<span class="pl-c">*/</span></span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">//</span> See docs in ../ops/array_ops.cc.</span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">define</span> <span class="pl-en">EIGEN_USE_THREADS</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&lt;</span>math.h<span class="pl-pds">&gt;</span></span></td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>third_party/eigen3/unsupported/Eigen/CXX11/Tensor<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/framework/op.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/framework/op_kernel.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/framework/type_traits.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/framework/types.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/kernels/meta_support.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/lib/core/errors.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/kernels/no_op.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">include</span> <span class="pl-s"><span class="pl-pds">&quot;</span>tensorflow/core/util/mkl_util.h<span class="pl-pds">&quot;</span></span></td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">ifdef</span> INTEL_MKL</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line"><span class="pl-k">namespace</span> <span class="pl-en">tensorflow</span> {</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line"><span class="pl-k">typedef</span> Eigen::ThreadPoolDevice CPUDevice;</td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">MklRequantizationRangePerChannelOp</span> : <span class="pl-k">public</span> <span class="pl-en">OpKernel</span> {</td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line"> <span class="pl-k">public:</span></td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">explicit</span> <span class="pl-en">MklRequantizationRangePerChannelOp</span>(OpKernelConstruction* ctx) : OpKernel(ctx) {</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">OP_REQUIRES_OK</span>(ctx, ctx-&gt;<span class="pl-c1">GetAttr</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>is_relu6<span class="pl-pds">&quot;</span></span>, &amp;is_relu6_));</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">  }</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">void</span> <span class="pl-en">Compute</span>(OpKernelContext* ctx) <span class="pl-k">override</span> {</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">const</span> Tensor&amp; input = <span class="pl-c1">MklGetInput</span>(ctx, <span class="pl-c1">kInputTensorIndex</span>);</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">const</span> Tensor&amp; input_min = <span class="pl-c1">MklGetInput</span>(ctx, <span class="pl-c1">this</span>-&gt;<span class="pl-smi">kInputMin</span>);</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">const</span> Tensor&amp; input_max = <span class="pl-c1">MklGetInput</span>(ctx, <span class="pl-c1">this</span>-&gt;<span class="pl-smi">kInputMax</span>);</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">size_t</span> depth = input_max.<span class="pl-c1">NumElements</span>();</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">OP_REQUIRES</span>(ctx, input_min.<span class="pl-c1">dim_size</span>(<span class="pl-c1">0</span>) == depth,</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">errors::InvalidArgument</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>min has incorrect size, expected <span class="pl-pds">&quot;</span></span>, depth,</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">                                <span class="pl-s"><span class="pl-pds">&quot;</span> was <span class="pl-pds">&quot;</span></span>, input_min.<span class="pl-c1">dim_size</span>(<span class="pl-c1">0</span>)));</td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">OP_REQUIRES</span>(ctx, input_max.<span class="pl-c1">dim_size</span>(<span class="pl-c1">0</span>) == depth,</td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">errors::InvalidArgument</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>max has incorrect size, expected <span class="pl-pds">&quot;</span></span>, depth,</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">                                <span class="pl-s"><span class="pl-pds">&quot;</span> was <span class="pl-pds">&quot;</span></span>, input_max.<span class="pl-c1">dim_size</span>(<span class="pl-c1">0</span>)));</td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">const</span> <span class="pl-k">float</span>* input_min_data = input_min.<span class="pl-smi">flat</span>&lt;<span class="pl-k">float</span>&gt;().<span class="pl-c1">data</span>();</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">const</span> <span class="pl-k">float</span>* input_max_data = input_max.<span class="pl-smi">flat</span>&lt;<span class="pl-k">float</span>&gt;().<span class="pl-c1">data</span>();</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">    std::vector&lt;<span class="pl-k">float</span>&gt; <span class="pl-c1">ranges</span>(depth);</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">bool</span> is_non_negative = <span class="pl-c1">true</span>;</td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">    Eigen::array&lt;<span class="pl-k">int</span>, <span class="pl-c1">2</span>&gt; <span class="pl-c1">shuffling</span>({<span class="pl-c1">1</span>, <span class="pl-c1">0</span>});</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">auto</span> input_matrix = input.<span class="pl-smi">flat_inner_dims</span>&lt;qint32&gt;();</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">auto</span> transposed_input = input_matrix.<span class="pl-c1">shuffle</span>(shuffling);</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">pragma</span> omp parallel for</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> (<span class="pl-c1">size_t</span> i = <span class="pl-c1">0</span>; i &lt; depth; i++)</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">    {</td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">      Eigen::Tensor&lt;qint32, <span class="pl-c1">0</span>, Eigen::RowMajor&gt; min =</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">          transposed_input.<span class="pl-smi">chip</span>&lt;<span class="pl-c1">0</span>&gt;(i).<span class="pl-c1">minimum</span>();</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">      Eigen::Tensor&lt;qint32, <span class="pl-c1">0</span>, Eigen::RowMajor&gt; max =</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">          transposed_input.<span class="pl-smi">chip</span>&lt;<span class="pl-c1">0</span>&gt;(i).<span class="pl-c1">maximum</span>();</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">      <span class="pl-c1">int32_t</span> min_per_channel = <span class="pl-c1">min</span>();</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">      <span class="pl-c1">int32_t</span> max_per_channel = <span class="pl-c1">max</span>();</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">      <span class="pl-c1">int32_t</span> abs_max = <span class="pl-c1">std::max</span>(<span class="pl-c1">std::abs</span>(min_per_channel), <span class="pl-c1">std::abs</span>(max_per_channel));</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">float</span> scale = <span class="pl-c1">std::max</span>(<span class="pl-c1">std::abs</span>(input_min_data[i]), <span class="pl-c1">std::abs</span>(input_max_data[i]));</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">      ranges[i] = (scale * (<span class="pl-k">float</span>)abs_max / (<span class="pl-k">float</span>)(<span class="pl-c1">1L</span>&lt;&lt;<span class="pl-c1">31</span>));</td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (min_per_channel &lt; <span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">        is_non_negative = <span class="pl-c1">false</span>;</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">float</span> out_min_max = std::numeric_limits&lt;<span class="pl-k">float</span>&gt;::<span class="pl-c1">min</span>();</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> (<span class="pl-c1">size_t</span> i = <span class="pl-c1">0</span>; i &lt; depth; i++) {</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">      <span class="pl-k">if</span> (out_min_max &lt; ranges[i])</td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">        out_min_max = ranges[i];</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">    } </td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">//</span> Fixing max to 6.0 for relu6</span></td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> (is_relu6_ &amp;&amp; out_min_max &gt; <span class="pl-c1">6</span>.<span class="pl-c1">0f</span>)</td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">      out_min_max = <span class="pl-c1">6</span>.<span class="pl-c1">0f</span>;</td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">    Tensor* output_min = <span class="pl-c1">nullptr</span>;</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">    Tensor* output_max = <span class="pl-c1">nullptr</span>;</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">OP_REQUIRES_OK</span>(ctx, ctx-&gt;<span class="pl-c1">allocate_output</span>(<span class="pl-c1">kOutputMin</span>, {}, &amp;output_min));</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">OP_REQUIRES_OK</span>(ctx, ctx-&gt;<span class="pl-c1">allocate_output</span>(<span class="pl-c1">kOutputMax</span>, {}, &amp;output_max));</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">    output_min-&gt;<span class="pl-smi">flat</span>&lt;<span class="pl-k">float</span>&gt;()(<span class="pl-c1">0</span>) = is_non_negative ? <span class="pl-c1">0</span>.<span class="pl-c1">0f</span> : out_min_max * -<span class="pl-c1">1</span>.<span class="pl-c1">0f</span>;</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">    output_max-&gt;<span class="pl-smi">flat</span>&lt;<span class="pl-k">float</span>&gt;()(<span class="pl-c1">0</span>) = out_min_max;</td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">  }</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">private:</span></td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">bool</span> is_relu6_ = <span class="pl-c1">false</span>;</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">const</span> <span class="pl-k">int</span> <span class="pl-c1">kInputTensorIndex</span>  = <span class="pl-c1">0</span>;</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">const</span> <span class="pl-k">int</span> <span class="pl-c1">kInputMin</span>          = <span class="pl-c1">1</span>;</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">const</span> <span class="pl-k">int</span> <span class="pl-c1">kInputMax</span>          = <span class="pl-c1">2</span>;</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">const</span> <span class="pl-k">int</span> <span class="pl-c1">kOutputMin</span>         = <span class="pl-c1">0</span>;</td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">const</span> <span class="pl-k">int</span> <span class="pl-c1">kOutputMax</span>         = <span class="pl-c1">1</span>;</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">};</td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line"><span class="pl-en">REGISTER_KERNEL_BUILDER</span>(Name(<span class="pl-s"><span class="pl-pds">&quot;</span>RequantizationRangePerChannel<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">                            .Device(DEVICE_CPU)</td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">                            .TypeConstraint&lt;qint32&gt;(<span class="pl-s"><span class="pl-pds">&quot;</span>T<span class="pl-pds">&quot;</span></span>),</td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">                        MklRequantizationRangePerChannelOp);</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">//</span> REGISTER_KERNEL_BUILDER(Name(&quot;RequantizationRangePerChannel&quot;)</span></td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">//</span>                             .Device(DEVICE_CPU)</span></td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">//</span>                             .TypeConstraint&lt;qint32&gt;(&quot;T&quot;),</span></td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">//</span>                         NoOp);</span></td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">}  <span class="pl-c"><span class="pl-c">//</span> namespace tensorflow</span></td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">#<span class="pl-k">endif</span> <span class="pl-c"><span class="pl-c">//</span> INTEL_MKL</span></td>
      </tr>
</table>

  <details class="details-reset details-overlay BlobToolbar position-absolute js-file-line-actions dropdown d-none" aria-hidden="true">
    <summary class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1" aria-label="Inline file action toolbar">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 13 16" version="1.1" width="13" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm5 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM13 7.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"/></svg>
    </summary>
    <details-menu>
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2">
        <li><clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-lines" style="cursor:pointer;" data-original-text="Copy lines">Copy lines</clipboard-copy></li>
        <li><clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;" data-original-text="Copy permalink">Copy permalink</clipboard-copy></li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" role="menuitem" href="/NervanaSystems/private-tensorflow/blame/02faed6a53512187bd02b1a7ce23646ff4df4d59/tensorflow/core/kernels/mkl_requantization_range_per_channel_op.cc">View git blame</a></li>
      </ul>
    </details-menu>
  </details>

  </div>

    </div>

  

  <details class="details-reset details-overlay details-overlay-dark">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>



  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>

    </div>
  </div>

  </div>

        
<div class="footer container-lg px-3" role="contentinfo">
  <div class="position-relative d-flex flex-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap ">
      <li class="mr-3">&copy; 2018 <span title="0.32076s from unicorn-6fc7f5966b-s9982">GitHub</span>, Inc.</li>
        <li class="mr-3"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3"><a href="/security" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li class="mr-3"><a href="https://githubstatus.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>
    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon mr-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap ">
        <li class="mr-3"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3"><a href="https://blog.github.com" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>

    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    <script crossorigin="anonymous" integrity="sha512-WnyO4VoIUwWWQOmFLjYf4UGg/c1z9VlaLN8IMuiI3uMhhl6rejyThRdLPDyePeUPW6N+38OoBMs6AkqcvWALtA==" type="application/javascript" src="https://github.githubassets.com/assets/compat-b66b5d97b4442a01f057c74b091c4368.js"></script>
    <script crossorigin="anonymous" integrity="sha512-qj1Z4g+YDVWKElb6FC42CcUWNq8P0W9aiU0CtKWjzuEMO9tb8sbbJ48dWUkmaOIe7WlLJXkI4YsUCKMxRMxT5A==" type="application/javascript" src="https://github.githubassets.com/assets/frameworks-9ee26246cce2c45ef24accded28cdabe.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-H0kSY4ldXVnWQwlnI0SeLKAmtmjXlpMc/QzrMKxoHhKIC2PiyWQtLI2ksLOSwzcVmrwR7vjiSzecbut6nHPxGw==" type="application/javascript" src="https://github.githubassets.com/assets/github-d30ac661d211116eedd9d8d55c71f002.js"></script>
    
    
    
  <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
  </div>
</div>

  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark" open>
    <summary aria-haspopup="dialog" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>

<div id="hovercard-aria-description" class="sr-only">
  Press h to open a hovercard with more details.
</div>

  <div aria-live="polite" class="js-global-screen-reader-notice sr-only"></div>

  </body>
</html>

